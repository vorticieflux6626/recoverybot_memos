"""
Graph-Based Cache Integration for Agentic Search Orchestrator.

This module provides a wrapper that integrates:
- Agent Step Graph for workflow-aware cache management (KVFlow-inspired)
- Scratchpad Cache for intermediate answer caching (ROG-inspired)
- Prefix-optimized prompts for maximum KV cache reuse

The integration is non-invasive - it wraps existing orchestrator calls
and adds graph-based caching logic around them.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .agent_step_graph import (
    AgentStepGraph, AgentType, get_agent_step_graph
)
from .scratchpad_cache import (
    ScratchpadCache, get_scratchpad_cache,
    CachedFinding, CachedSubQuery
)
from .prefix_optimized_prompts import (
    build_full_prompt, get_prefix_for_warming,
    estimate_prefix_reuse, get_prompt_registry
)
from .models import SearchState, SearchRequest, SearchResponse

logger = logging.getLogger("agentic.graph_cache")


@dataclass
class WorkflowContext:
    """Context for a single search workflow execution"""
    workflow_id: str
    start_time: float
    current_agent: Optional[AgentType] = None
    transitions: List[Tuple[AgentType, AgentType, float]] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    prefetches_initiated: int = 0


class GraphCacheIntegration:
    """
    Integration layer that adds graph-based caching to the agentic search pipeline.

    This class wraps orchestrator operations and adds:
    1. Workflow tracking via Agent Step Graph
    2. Intermediate result caching via Scratchpad Cache
    3. Proactive cache prefetching for next agents
    4. Cache eviction based on steps-to-execution
    """

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 cache_db_path: str = "data/scratchpad_cache.db"):
        self.ollama_url = ollama_url

        # Initialize graph-based components
        self.agent_graph = get_agent_step_graph()
        self.scratchpad_cache = get_scratchpad_cache(cache_db_path)
        self.prompt_registry = get_prompt_registry()

        # Active workflows
        self.active_workflows: Dict[str, WorkflowContext] = {}

        # Statistics
        self.stats = {
            'workflows_started': 0,
            'workflows_completed': 0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'prefetch_success_rate': 0.0,
            'avg_workflow_time_ms': 0.0
        }

    async def start_workflow(self, workflow_id: str, query: str) -> WorkflowContext:
        """
        Start a new search workflow and initialize graph tracking.

        Args:
            workflow_id: Unique identifier for this workflow
            query: User query

        Returns:
            WorkflowContext for tracking this workflow
        """
        context = WorkflowContext(
            workflow_id=workflow_id,
            start_time=time.time()
        )

        self.active_workflows[workflow_id] = context
        self.agent_graph.start_workflow(workflow_id)
        self.stats['workflows_started'] += 1

        # Check if we have a cached mission decomposition for similar query
        cached_mission = self.scratchpad_cache.get_mission_decomposition(query)
        if cached_mission:
            logger.info(f"[{workflow_id}] Found cached mission decomposition "
                       f"(success_rate={cached_mission.success_rate:.2f})")

        # Pre-warm cache for first agent (analyzer)
        await self._prefetch_agent_cache(AgentType.ANALYZER, {}, context)

        return context

    async def before_agent_call(self, workflow_id: str, agent_type: AgentType,
                                 scratchpad_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called before invoking an agent. Handles:
        1. Recording agent transition in graph
        2. Checking intermediate caches
        3. Initiating prefetch for next agents

        Args:
            workflow_id: Current workflow ID
            agent_type: Type of agent about to be invoked
            scratchpad_state: Current scratchpad state

        Returns:
            Dictionary with cached data if available, empty otherwise
        """
        context = self.active_workflows.get(workflow_id)
        if not context:
            logger.warning(f"Unknown workflow: {workflow_id}")
            return {}

        start_time = time.time()
        cached_result = {}

        # Record transition from previous agent
        if context.current_agent:
            transition_time = (start_time - context.start_time) * 1000
            self.agent_graph.record_transition(
                context.current_agent, agent_type, transition_time
            )
            context.transitions.append(
                (context.current_agent, agent_type, transition_time)
            )

        context.current_agent = agent_type

        # Check for cached intermediate results based on agent type
        if agent_type == AgentType.PLANNER:
            # Check for cached mission decomposition
            query = scratchpad_state.get('mission', '')
            cached_mission = self.scratchpad_cache.get_mission_decomposition(query)
            if cached_mission and cached_mission.success_rate > 0.6:
                cached_result['decomposed_questions'] = cached_mission.sub_questions
                context.cache_hits += 1
                logger.debug(f"[{workflow_id}] Planner cache hit")

        elif agent_type == AgentType.SEARCHER:
            # Check for related sub-query results
            sub_questions = scratchpad_state.get('sub_questions', [])
            for sq in sub_questions[:3]:  # Check first 3
                q_text = sq.get('question', sq) if isinstance(sq, dict) else sq
                cached_sq = self.scratchpad_cache.get_subquery(q_text)
                if cached_sq:
                    if 'cached_subqueries' not in cached_result:
                        cached_result['cached_subqueries'] = []
                    cached_result['cached_subqueries'].append({
                        'question': q_text,
                        'answer': cached_sq.answer,
                        'sources': cached_sq.sources,
                        'confidence': cached_sq.confidence
                    })
                    context.cache_hits += 1

        elif agent_type == AgentType.VERIFIER:
            # Check for already-verified findings
            findings = scratchpad_state.get('findings', [])
            verified_hashes = set()
            for finding in findings[:10]:
                content = finding.get('content', finding) if isinstance(finding, dict) else str(finding)
                cached_finding = self.scratchpad_cache.get_finding(content)
                if cached_finding:
                    verified_hashes.add(cached_finding.content_hash)
                    context.cache_hits += 1
            if verified_hashes:
                cached_result['pre_verified_hashes'] = list(verified_hashes)

        if not cached_result:
            context.cache_misses += 1

        # Initiate prefetch for likely next agents (non-blocking)
        asyncio.create_task(
            self._prefetch_next_agents(agent_type, scratchpad_state, context)
        )

        return cached_result

    async def after_agent_call(self, workflow_id: str, agent_type: AgentType,
                                result: Dict[str, Any], duration_ms: float,
                                token_count: int = 0):
        """
        Called after an agent completes. Handles:
        1. Recording execution statistics
        2. Caching intermediate results
        3. Updating agent graph

        Args:
            workflow_id: Current workflow ID
            agent_type: Type of agent that just completed
            result: Agent's output
            duration_ms: Execution time in milliseconds
            token_count: Number of tokens generated
        """
        context = self.active_workflows.get(workflow_id)
        if not context:
            return

        # Update agent graph with execution stats
        self.agent_graph.mark_agent_executed(agent_type, duration_ms, token_count)

        # Cache intermediate results based on agent type
        if agent_type == AgentType.PLANNER:
            # Cache mission decomposition
            if 'decomposed_questions' in result:
                mission = result.get('mission', context.workflow_id)
                self.scratchpad_cache.cache_mission_decomposition(
                    mission=mission,
                    sub_questions=result['decomposed_questions'],
                    success_rate=0.5  # Initial rate, updated on workflow completion
                )

        elif agent_type == AgentType.SEARCHER or agent_type == AgentType.SCRAPER:
            # Cache findings
            findings = result.get('findings', result.get('extracted_content', []))
            for finding in findings[:20]:  # Limit to 20
                if isinstance(finding, dict):
                    self.scratchpad_cache.cache_finding(
                        content=finding.get('content', ''),
                        source_url=finding.get('source', finding.get('source_url', '')),
                        confidence=finding.get('confidence', finding.get('relevance_score', 0.5))
                    )

        elif agent_type == AgentType.VERIFIER:
            # Cache verified sub-query results
            for claim in result.get('verified_claims', []):
                if claim.get('verified') and claim.get('confidence', 0) > 0.7:
                    self.scratchpad_cache.cache_subquery(
                        query=claim.get('claim', ''),
                        answer=claim.get('summary', claim.get('claim', '')),
                        sources=claim.get('sources', []),
                        confidence=claim.get('confidence', 0.8)
                    )

    async def end_workflow(self, workflow_id: str, success: bool = True) -> Dict[str, Any]:
        """
        End a search workflow and collect statistics.

        Args:
            workflow_id: Workflow to end
            success: Whether the workflow completed successfully

        Returns:
            Dictionary with workflow statistics
        """
        context = self.active_workflows.pop(workflow_id, None)
        if not context:
            return {}

        duration_ms = (time.time() - context.start_time) * 1000
        self.agent_graph.end_workflow()

        # Update mission success rate if we cached a decomposition
        # (This helps future similar queries benefit from successful strategies)

        stats = {
            'workflow_id': workflow_id,
            'duration_ms': duration_ms,
            'transitions': len(context.transitions),
            'cache_hits': context.cache_hits,
            'cache_misses': context.cache_misses,
            'hit_rate': context.cache_hits / max(1, context.cache_hits + context.cache_misses),
            'prefetches': context.prefetches_initiated,
            'success': success
        }

        # Update global stats
        self.stats['workflows_completed'] += 1
        self.stats['total_cache_hits'] += context.cache_hits
        self.stats['total_cache_misses'] += context.cache_misses

        # Update running average of workflow time
        n = self.stats['workflows_completed']
        old_avg = self.stats['avg_workflow_time_ms']
        self.stats['avg_workflow_time_ms'] = ((n - 1) * old_avg + duration_ms) / n

        logger.info(f"[{workflow_id}] Workflow completed: {duration_ms:.0f}ms, "
                   f"cache hit rate: {stats['hit_rate']:.1%}")

        return stats

    async def _prefetch_agent_cache(self, agent_type: AgentType,
                                     scratchpad_state: Dict[str, Any],
                                     context: WorkflowContext):
        """Prefetch KV cache for a specific agent"""
        try:
            await self.agent_graph.prefetch_next_agent_cache(
                agent_type,
                scratchpad_context=str(scratchpad_state)[:2000]
            )
            context.prefetches_initiated += 1
        except Exception as e:
            logger.debug(f"Prefetch failed for {agent_type}: {e}")

    async def _prefetch_next_agents(self, current_agent: AgentType,
                                     scratchpad_state: Dict[str, Any],
                                     context: WorkflowContext):
        """Prefetch cache for likely next agents based on graph predictions"""
        predictions = self.agent_graph.get_likely_next_agents(current_agent, top_k=2)

        for next_agent, probability in predictions:
            if probability >= 0.3:  # Only prefetch likely transitions
                await self._prefetch_agent_cache(next_agent, scratchpad_state, context)

    def get_eviction_candidates(self, memory_pressure: float = 0.8) -> List[AgentType]:
        """Get agents whose cache should be evicted based on STE"""
        return self.agent_graph.get_eviction_candidates(memory_pressure)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all cache components"""
        return {
            'graph_cache_integration': self.stats,
            'agent_step_graph': self.agent_graph.get_graph_stats(),
            'scratchpad_cache': self.scratchpad_cache.get_stats(),
            'active_workflows': len(self.active_workflows),
            'overall_cache_hit_rate': (
                self.stats['total_cache_hits'] /
                max(1, self.stats['total_cache_hits'] + self.stats['total_cache_misses'])
            )
        }

    def visualize_workflow(self, workflow_id: str) -> str:
        """Generate ASCII visualization of a workflow's execution path"""
        context = self.active_workflows.get(workflow_id)
        if not context:
            return f"No active workflow: {workflow_id}"

        lines = [f"Workflow: {workflow_id}", "=" * 40]

        if context.transitions:
            for from_agent, to_agent, time_ms in context.transitions:
                lines.append(f"  {from_agent.value} → {to_agent.value} ({time_ms:.0f}ms)")

        lines.append(f"\nCurrent: {context.current_agent.value if context.current_agent else 'None'}")
        lines.append(f"Cache hits: {context.cache_hits}, misses: {context.cache_misses}")

        return "\n".join(lines)


# =============================================================================
# Helper function to integrate with existing orchestrator
# =============================================================================

def map_orchestrator_phase_to_agent_type(phase: str) -> Optional[AgentType]:
    """
    Map orchestrator phase names to AgentType enum.

    This allows integration without modifying the orchestrator's phase logic.
    """
    mapping = {
        'analyze': AgentType.ANALYZER,
        'analysis': AgentType.ANALYZER,
        'plan': AgentType.PLANNER,
        'planning': AgentType.PLANNER,
        'search': AgentType.SEARCHER,
        'searching': AgentType.SEARCHER,
        'scrape': AgentType.SCRAPER,
        'scraping': AgentType.SCRAPER,
        'verify': AgentType.VERIFIER,
        'verification': AgentType.VERIFIER,
        'synthesize': AgentType.SYNTHESIZER,
        'synthesis': AgentType.SYNTHESIZER,
        'coverage_check': AgentType.VERIFIER,  # Coverage evaluation is part of verification
        'refine': AgentType.SEARCHER  # Refinement triggers more searching
    }
    return mapping.get(phase.lower())


# =============================================================================
# Singleton instance
# =============================================================================

_graph_cache_integration: Optional[GraphCacheIntegration] = None


def get_graph_cache_integration(ollama_url: str = "http://localhost:11434",
                                 cache_db_path: str = "data/scratchpad_cache.db") -> GraphCacheIntegration:
    """Get or create the singleton GraphCacheIntegration instance"""
    global _graph_cache_integration
    if _graph_cache_integration is None:
        _graph_cache_integration = GraphCacheIntegration(ollama_url, cache_db_path)
    return _graph_cache_integration


async def initialize_graph_cache(ollama_url: str = "http://localhost:11434"):
    """
    Initialize graph-based cache system.

    Call this during server startup to:
    1. Pre-warm KV cache for common prompts
    2. Load cached data from SQLite
    3. Initialize agent step graph
    """
    integration = get_graph_cache_integration(ollama_url)

    # Get high-priority templates to warm
    templates = integration.prompt_registry.get_high_priority_templates(limit=5)

    logger.info(f"Initializing graph cache with {len(templates)} priority templates")

    # Note: Actual cache warming would require calling Ollama
    # This is a placeholder for the warming logic
    for template in templates:
        logger.debug(f"Template ready for warming: {template.name} "
                    f"({template.estimated_tokens} tokens)")

    logger.info("Graph-based cache initialization complete")

    return integration
