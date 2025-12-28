"""
Agent Step Graph - KVFlow-inspired workflow-aware KV cache management.

Based on: Pan et al., "Efficient Prefix Caching for Accelerating LLM-Based
Multi-Agent Workflows", NeurIPS 2025

Key concepts:
- Agent Step Graph: DAG representing workflow dependencies
- Steps-to-Execution (STE): Distance metric for cache eviction priority
- Proactive Prefetching: Load KV cache before agent activation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import asyncio
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent types in the memOS agentic search workflow"""
    ORCHESTRATOR = "orchestrator"
    ANALYZER = "analyzer"
    PLANNER = "planner"
    SEARCHER = "searcher"
    SCRAPER = "scraper"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"


@dataclass
class AgentNode:
    """Node in the Agent Step Graph"""
    agent_type: AgentType
    prompt_prefix: str  # Static portion of prompt for this agent
    dependencies: List[AgentType] = field(default_factory=list)
    successors: List[AgentType] = field(default_factory=list)
    steps_to_execution: int = 0
    kv_cache_id: Optional[str] = None
    last_access_time: float = 0.0
    estimated_duration_ms: float = 0.0
    execution_count: int = 0
    avg_token_count: int = 0

    def update_stats(self, duration_ms: float, token_count: int):
        """Update running statistics for this agent"""
        self.execution_count += 1
        # Exponential moving average
        alpha = 0.3
        self.estimated_duration_ms = (
            alpha * duration_ms + (1 - alpha) * self.estimated_duration_ms
            if self.estimated_duration_ms > 0 else duration_ms
        )
        self.avg_token_count = int(
            alpha * token_count + (1 - alpha) * self.avg_token_count
            if self.avg_token_count > 0 else token_count
        )


@dataclass
class GraphEdge:
    """Edge in the Agent Step Graph representing transition probability"""
    from_agent: AgentType
    to_agent: AgentType
    transition_count: int = 0
    avg_transition_time_ms: float = 0.0

    @property
    def transition_probability(self) -> float:
        """Calculate transition probability based on historical data"""
        # Will be normalized across all outgoing edges
        return self.transition_count


class AgentStepGraph:
    """
    KVFlow-inspired Agent Step Graph for workflow-aware cache management.

    Each agent has a steps-to-execution (STE) value:
    - Lower STE = higher priority for cache retention (about to execute)
    - Higher STE = candidate for eviction under memory pressure

    The graph learns from execution patterns to optimize prefetching.
    """

    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        self.nodes: Dict[AgentType, AgentNode] = {}
        self.edges: Dict[Tuple[AgentType, AgentType], GraphEdge] = {}
        self.execution_history: List[AgentType] = []
        self.current_workflow_id: Optional[str] = None

        # Statistics
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.eviction_count = 0

        # Build the default workflow graph
        self._build_default_graph()

    def _build_default_graph(self):
        """
        Build the default memOS agentic search workflow graph.

        Workflow structure:

        ORCHESTRATOR → ANALYZER → PLANNER
                                    ↓
                                SEARCHER ←→ SCRAPER
                                    ↓
                                VERIFIER
                                    ↓
                               SYNTHESIZER
        """
        # Create nodes with agent-specific prompt prefixes
        self.nodes = {
            AgentType.ORCHESTRATOR: AgentNode(
                agent_type=AgentType.ORCHESTRATOR,
                prompt_prefix=self._get_orchestrator_prefix(),
                successors=[AgentType.ANALYZER],
                steps_to_execution=0
            ),
            AgentType.ANALYZER: AgentNode(
                agent_type=AgentType.ANALYZER,
                prompt_prefix=self._get_analyzer_prefix(),
                dependencies=[AgentType.ORCHESTRATOR],
                successors=[AgentType.PLANNER],
                steps_to_execution=1
            ),
            AgentType.PLANNER: AgentNode(
                agent_type=AgentType.PLANNER,
                prompt_prefix=self._get_planner_prefix(),
                dependencies=[AgentType.ANALYZER],
                successors=[AgentType.SEARCHER],
                steps_to_execution=2
            ),
            AgentType.SEARCHER: AgentNode(
                agent_type=AgentType.SEARCHER,
                prompt_prefix=self._get_searcher_prefix(),
                dependencies=[AgentType.PLANNER],
                successors=[AgentType.SCRAPER, AgentType.VERIFIER],
                steps_to_execution=3
            ),
            AgentType.SCRAPER: AgentNode(
                agent_type=AgentType.SCRAPER,
                prompt_prefix=self._get_scraper_prefix(),
                dependencies=[AgentType.SEARCHER],
                successors=[AgentType.SEARCHER, AgentType.VERIFIER],
                steps_to_execution=3  # Same level as searcher (can loop)
            ),
            AgentType.VERIFIER: AgentNode(
                agent_type=AgentType.VERIFIER,
                prompt_prefix=self._get_verifier_prefix(),
                dependencies=[AgentType.SEARCHER, AgentType.SCRAPER],
                successors=[AgentType.SYNTHESIZER],
                steps_to_execution=4
            ),
            AgentType.SYNTHESIZER: AgentNode(
                agent_type=AgentType.SYNTHESIZER,
                prompt_prefix=self._get_synthesizer_prefix(),
                dependencies=[AgentType.VERIFIER],
                successors=[],
                steps_to_execution=5
            )
        }

        # Initialize edges with default transitions
        default_transitions = [
            (AgentType.ORCHESTRATOR, AgentType.ANALYZER),
            (AgentType.ANALYZER, AgentType.PLANNER),
            (AgentType.PLANNER, AgentType.SEARCHER),
            (AgentType.SEARCHER, AgentType.SCRAPER),
            (AgentType.SCRAPER, AgentType.VERIFIER),
            (AgentType.SEARCHER, AgentType.VERIFIER),
            (AgentType.VERIFIER, AgentType.SYNTHESIZER),
            # Loop edges for iterative refinement
            (AgentType.SCRAPER, AgentType.SEARCHER),
        ]

        for from_agent, to_agent in default_transitions:
            self.edges[(from_agent, to_agent)] = GraphEdge(
                from_agent=from_agent,
                to_agent=to_agent,
                transition_count=1  # Initial prior
            )

    # --- Prompt Prefixes (Static, cacheable portions) ---

    def _get_orchestrator_prefix(self) -> str:
        return """SYSTEM: You are the Orchestrator Agent in an intelligent agentic search system.
Your role is to receive user queries and initiate the search workflow.

RESPONSIBILITIES:
- Analyze query intent and complexity
- Determine if agentic search is needed vs simple lookup
- Initialize workflow state

OUTPUT FORMAT: JSON with fields: requires_search, complexity, initial_context"""

    def _get_analyzer_prefix(self) -> str:
        return """SYSTEM: You are the Analyzer Agent in an intelligent agentic search system.
Your role is to analyze queries and determine search requirements.

RESPONSIBILITIES:
- Classify query type (factual, comparative, procedural, technical, troubleshooting)
- Identify key entities and concepts
- Assess information needs

OUTPUT FORMAT: JSON with fields: query_type, complexity, requires_search, reasoning"""

    def _get_planner_prefix(self) -> str:
        return """SYSTEM: You are the Planner Agent in an intelligent agentic search system.
Your role is to decompose complex queries into searchable sub-questions.

RESPONSIBILITIES:
- Break down complex queries into atomic sub-questions
- Define completion criteria for each sub-question
- Prioritize sub-questions by importance

OUTPUT FORMAT: JSON with fields: decomposed_questions (list with question, criteria, priority)"""

    def _get_searcher_prefix(self) -> str:
        return """SYSTEM: You are the Searcher Agent in an intelligent agentic search system.
Your role is to execute web searches and manage search results.

RESPONSIBILITIES:
- Generate effective search queries
- Evaluate URL relevance before scraping
- Track search coverage

OUTPUT FORMAT: JSON with fields: search_queries, url_evaluations, coverage_score"""

    def _get_scraper_prefix(self) -> str:
        return """SYSTEM: You are the Scraper Agent in an intelligent agentic search system.
Your role is to extract and process content from web pages.

RESPONSIBILITIES:
- Extract relevant content from URLs
- Structure extracted information
- Identify information gaps

OUTPUT FORMAT: JSON with fields: extracted_content, source_url, relevance_score"""

    def _get_verifier_prefix(self) -> str:
        return """SYSTEM: You are the Verifier Agent in an intelligent agentic search system.
Your role is to cross-check facts and assess information confidence.

RESPONSIBILITIES:
- Verify claims across multiple sources
- Detect contradictions
- Calculate confidence scores

OUTPUT FORMAT: JSON with fields: verified_claims, contradictions, confidence_score"""

    def _get_synthesizer_prefix(self) -> str:
        return """SYSTEM: You are the Synthesizer Agent in an intelligent agentic search system.
Your role is to combine verified findings into a coherent response.

RESPONSIBILITIES:
- Synthesize information from all sources
- Add inline citations [Source X]
- Structure response for clarity

OUTPUT FORMAT: Markdown with inline citations and source list at the end"""

    # --- Graph Traversal Methods ---

    def compute_steps_to_execution(self, current_agent: AgentType) -> Dict[AgentType, int]:
        """
        Compute STE values from current execution point using BFS.

        Returns dict mapping each agent to its steps-to-execution value.
        Lower values = higher priority for cache retention.
        """
        ste_values = {}
        visited = set()
        queue = [(current_agent, 0)]

        # Forward BFS to compute distance to future agents
        while queue:
            agent, distance = queue.pop(0)
            if agent in visited:
                continue
            visited.add(agent)
            ste_values[agent] = distance

            node = self.nodes.get(agent)
            if node:
                for successor in node.successors:
                    if successor not in visited:
                        queue.append((successor, distance + 1))

        # Agents not in forward path get high STE (eviction candidates)
        for agent_type in self.nodes.keys():
            if agent_type not in ste_values:
                # Calculate reverse distance (how long ago executed)
                if agent_type in self.execution_history:
                    reverse_idx = len(self.execution_history) - 1 - \
                                  self.execution_history[::-1].index(agent_type)
                    ste_values[agent_type] = len(self.nodes) + \
                                             (len(self.execution_history) - reverse_idx)
                else:
                    ste_values[agent_type] = len(self.nodes) * 2

        return ste_values

    def get_likely_next_agents(self, current_agent: AgentType,
                                top_k: int = 2) -> List[Tuple[AgentType, float]]:
        """
        Predict most likely next agents based on transition probabilities.

        Returns list of (agent_type, probability) tuples.
        """
        node = self.nodes.get(current_agent)
        if not node or not node.successors:
            return []

        # Calculate total transitions from current agent
        total_transitions = sum(
            self.edges.get((current_agent, succ), GraphEdge(current_agent, succ)).transition_count
            for succ in node.successors
        )

        if total_transitions == 0:
            # Uniform distribution
            prob = 1.0 / len(node.successors)
            return [(succ, prob) for succ in node.successors[:top_k]]

        # Calculate probabilities
        predictions = []
        for successor in node.successors:
            edge = self.edges.get((current_agent, successor))
            if edge:
                prob = edge.transition_count / total_transitions
                predictions.append((successor, prob))

        # Sort by probability descending
        predictions.sort(key=lambda x: -x[1])
        return predictions[:top_k]

    def record_transition(self, from_agent: AgentType, to_agent: AgentType,
                          transition_time_ms: float):
        """Record an observed agent transition to update the graph."""
        edge_key = (from_agent, to_agent)

        if edge_key in self.edges:
            edge = self.edges[edge_key]
            edge.transition_count += 1
            # Exponential moving average for transition time
            alpha = 0.3
            edge.avg_transition_time_ms = (
                alpha * transition_time_ms + (1 - alpha) * edge.avg_transition_time_ms
                if edge.avg_transition_time_ms > 0 else transition_time_ms
            )
        else:
            # New edge discovered (dynamic workflow pattern)
            self.edges[edge_key] = GraphEdge(
                from_agent=from_agent,
                to_agent=to_agent,
                transition_count=1,
                avg_transition_time_ms=transition_time_ms
            )
            # Update successors
            if from_agent in self.nodes:
                if to_agent not in self.nodes[from_agent].successors:
                    self.nodes[from_agent].successors.append(to_agent)

        self.execution_history.append(to_agent)

        # Keep history bounded
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-50:]

    # --- Cache Management Methods ---

    async def prefetch_next_agent_cache(self, current_agent: AgentType,
                                        scratchpad_context: str = ""):
        """
        Proactively load KV cache for likely next agents.
        This overlaps I/O with current agent's processing.
        """
        if not self.cache_manager:
            return

        predictions = self.get_likely_next_agents(current_agent, top_k=2)

        for next_agent, probability in predictions:
            if probability < 0.2:  # Don't prefetch unlikely paths
                continue

            node = self.nodes.get(next_agent)
            if not node:
                continue

            # Build prefix to warm
            prefix = node.prompt_prefix
            if scratchpad_context:
                prefix += f"\n\nCONTEXT:\n{scratchpad_context[:2000]}"  # Truncate if needed

            # Prefetch in background
            try:
                cache_id = await self.cache_manager.warm_prefix(prefix, next_agent.value)
                node.kv_cache_id = cache_id
                node.last_access_time = time.time()
                self.prefetch_hits += 1
                logger.debug(f"Prefetched cache for {next_agent.value} (prob={probability:.2f})")
            except Exception as e:
                self.prefetch_misses += 1
                logger.warning(f"Failed to prefetch cache for {next_agent.value}: {e}")

    def get_eviction_candidates(self, memory_pressure: float = 0.8) -> List[AgentType]:
        """
        Return agents whose KV cache can be evicted based on STE.

        Args:
            memory_pressure: 0.0-1.0 indicating memory usage level

        Returns:
            List of agents to evict, sorted by eviction priority
        """
        if not self.execution_history:
            return []

        current = self.execution_history[-1]
        ste_values = self.compute_steps_to_execution(current)

        # Sort by STE descending (highest = best eviction candidate)
        sorted_agents = sorted(ste_values.items(), key=lambda x: -x[1])

        # Calculate how many to evict based on memory pressure
        num_to_evict = max(1, int(len(sorted_agents) * (memory_pressure - 0.7) / 0.3))

        candidates = [agent for agent, _ in sorted_agents[:num_to_evict]]
        logger.debug(f"Eviction candidates (pressure={memory_pressure:.2f}): {candidates}")

        return candidates

    def mark_agent_executed(self, agent_type: AgentType,
                            duration_ms: float, token_count: int):
        """Record that an agent has executed and update statistics."""
        node = self.nodes.get(agent_type)
        if node:
            node.update_stats(duration_ms, token_count)
            node.last_access_time = time.time()

        # Record transition if we have history
        if self.execution_history:
            prev_agent = self.execution_history[-1]
            self.record_transition(prev_agent, agent_type, duration_ms)
        else:
            self.execution_history.append(agent_type)

    def start_workflow(self, workflow_id: str):
        """Start a new workflow, resetting execution history."""
        self.current_workflow_id = workflow_id
        self.execution_history = []

    def end_workflow(self):
        """End current workflow."""
        self.current_workflow_id = None

    # --- Statistics and Debugging ---

    def get_graph_stats(self) -> Dict:
        """Return statistics about the agent step graph."""
        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'execution_history_length': len(self.execution_history),
            'prefetch_hit_rate': (
                self.prefetch_hits / max(1, self.prefetch_hits + self.prefetch_misses)
            ),
            'prefetch_hits': self.prefetch_hits,
            'prefetch_misses': self.prefetch_misses,
            'eviction_count': self.eviction_count,
            'agent_stats': {
                agent.value: {
                    'execution_count': node.execution_count,
                    'avg_duration_ms': node.estimated_duration_ms,
                    'avg_tokens': node.avg_token_count
                }
                for agent, node in self.nodes.items()
            },
            'transition_stats': {
                f"{e.from_agent.value}->{e.to_agent.value}": {
                    'count': e.transition_count,
                    'avg_time_ms': e.avg_transition_time_ms
                }
                for e in self.edges.values()
            }
        }

    def visualize_graph(self) -> str:
        """Return ASCII visualization of the agent step graph."""
        lines = ["Agent Step Graph:", "=" * 40]

        for agent_type, node in self.nodes.items():
            successors = ", ".join(s.value for s in node.successors) or "None"
            lines.append(f"{agent_type.value} (STE={node.steps_to_execution})")
            lines.append(f"  → {successors}")
            if node.execution_count > 0:
                lines.append(f"  Stats: {node.execution_count} runs, "
                           f"avg {node.estimated_duration_ms:.0f}ms")

        return "\n".join(lines)


# Singleton instance
_agent_step_graph: Optional[AgentStepGraph] = None


def get_agent_step_graph(cache_manager=None) -> AgentStepGraph:
    """Get or create the singleton AgentStepGraph instance."""
    global _agent_step_graph
    if _agent_step_graph is None:
        _agent_step_graph = AgentStepGraph(cache_manager)
    elif cache_manager and _agent_step_graph.cache_manager is None:
        _agent_step_graph.cache_manager = cache_manager
    return _agent_step_graph
