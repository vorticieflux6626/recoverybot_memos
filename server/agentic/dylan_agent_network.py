"""
DyLAN Agent Importance Scores Module.

Based on DyLAN (Dynamic Language Agent Network) research (2025) achieving
25% accuracy improvement via conditional agent skipping.

Key insight: Not all agents need to run for every query. Simple queries
can skip verification/scraping agents when retrieval confidence is high.

Features:
- Agent importance scoring based on contribution tracking
- Query complexity classification
- Conditional agent skipping for simple queries
- Peer rating propagation via exponential moving average
- Warm-start priors for new queries

G.6.2 Implementation.

References:
- DyLAN: Dynamic Language Agent Network (2025)
- Agent Importance Score = unsupervised metric propagating peer ratings
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)


class QueryComplexity(str, Enum):
    """Query complexity levels for agent routing."""
    SIMPLE = "simple"          # Direct lookup, single concept
    MODERATE = "moderate"      # Multi-concept, some reasoning
    COMPLEX = "complex"        # Multi-hop, synthesis required
    EXPLORATORY = "exploratory"  # Open-ended, research-style


class AgentRole(str, Enum):
    """Roles of agents in the pipeline."""
    ANALYZER = "analyzer"           # Query analysis
    PLANNER = "planner"             # Search planning
    SEARCHER = "searcher"           # Web/doc search
    EVALUATOR = "evaluator"         # CRAG evaluation
    SCRAPER = "scraper"             # Web scraping
    VERIFIER = "verifier"           # Claim verification
    SYNTHESIZER = "synthesizer"     # Answer synthesis
    REFLECTOR = "reflector"         # Self-RAG reflection


# Default agent configuration
DEFAULT_AGENT_CONFIG = {
    AgentRole.ANALYZER: {
        "always_run": True,         # Always needed
        "base_importance": 1.0,
        "skip_threshold": None,     # Never skip
    },
    AgentRole.PLANNER: {
        "always_run": True,
        "base_importance": 1.0,
        "skip_threshold": None,
    },
    AgentRole.SEARCHER: {
        "always_run": True,
        "base_importance": 1.0,
        "skip_threshold": None,
    },
    AgentRole.EVALUATOR: {
        "always_run": False,
        "base_importance": 0.8,
        "skip_threshold": 0.85,     # Skip if retrieval confidence > 0.85
    },
    AgentRole.SCRAPER: {
        "always_run": False,
        "base_importance": 0.7,
        "skip_threshold": 0.90,     # Skip if synthesis confidence > 0.90
    },
    AgentRole.VERIFIER: {
        "always_run": False,
        "base_importance": 0.8,
        "skip_threshold": 0.85,     # Skip for simple queries
    },
    AgentRole.SYNTHESIZER: {
        "always_run": True,
        "base_importance": 1.0,
        "skip_threshold": None,
    },
    AgentRole.REFLECTOR: {
        "always_run": False,
        "base_importance": 0.7,
        "skip_threshold": 0.80,     # Skip if synthesis quality high
    },
}


@dataclass
class AgentContribution:
    """Tracks an agent's contribution to a search outcome."""
    agent_role: AgentRole
    execution_time_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    quality_delta: float = 0.0     # Change in quality after agent runs
    was_skipped: bool = False
    skip_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_role": self.agent_role.value,
            "execution_time_ms": self.execution_time_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "quality_delta": self.quality_delta,
            "was_skipped": self.was_skipped,
            "skip_reason": self.skip_reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentStats:
    """Aggregate statistics for an agent."""
    agent_role: AgentRole
    total_runs: int = 0
    total_skips: int = 0
    total_time_ms: float = 0.0
    total_quality_delta: float = 0.0
    importance_score: float = 1.0
    ema_alpha: float = 0.1          # EMA decay rate

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.total_runs, 1)

    @property
    def avg_quality_delta(self) -> float:
        return self.total_quality_delta / max(self.total_runs, 1)

    @property
    def skip_rate(self) -> float:
        total = self.total_runs + self.total_skips
        return self.total_skips / max(total, 1)

    def update_importance(self, contribution: AgentContribution) -> None:
        """Update importance score with EMA."""
        if not contribution.was_skipped:
            self.total_runs += 1
            self.total_time_ms += contribution.execution_time_ms
            self.total_quality_delta += contribution.quality_delta

            # EMA update: new_score = alpha * observed + (1 - alpha) * old_score
            # Quality delta normalized to [0, 1]
            observed_value = min(max(contribution.quality_delta + 0.5, 0), 1)
            self.importance_score = (
                self.ema_alpha * observed_value +
                (1 - self.ema_alpha) * self.importance_score
            )
        else:
            self.total_skips += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_role": self.agent_role.value,
            "total_runs": self.total_runs,
            "total_skips": self.total_skips,
            "avg_time_ms": round(self.avg_time_ms, 2),
            "avg_quality_delta": round(self.avg_quality_delta, 4),
            "importance_score": round(self.importance_score, 4),
            "skip_rate": round(self.skip_rate, 4),
        }


@dataclass
class SkipDecision:
    """Result of agent skip decision."""
    agent_role: AgentRole
    should_skip: bool
    reason: str
    confidence: float = 0.0
    alternative_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_role": self.agent_role.value,
            "should_skip": self.should_skip,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "alternative_action": self.alternative_action,
        }


@dataclass
class QueryComplexityResult:
    """Result of query complexity classification."""
    complexity: QueryComplexity
    confidence: float
    indicators: List[str]
    recommended_agents: List[AgentRole]
    skippable_agents: List[AgentRole]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "confidence": round(self.confidence, 3),
            "indicators": self.indicators,
            "recommended_agents": [a.value for a in self.recommended_agents],
            "skippable_agents": [a.value for a in self.skippable_agents],
        }


class DyLANAgentNetwork:
    """
    Dynamic Language Agent Network for conditional agent execution.

    Implements:
    1. Query complexity classification
    2. Agent importance scoring
    3. Conditional agent skipping
    4. Contribution tracking and learning

    Example:
        network = DyLANAgentNetwork()
        complexity = await network.classify_complexity(query)
        active_agents = network.get_active_agents(complexity, state)

        for agent in active_agents:
            result = await agent.execute(state)
            contribution = AgentContribution(...)
            network.record_contribution(contribution)
    """

    def __init__(
        self,
        agent_config: Optional[Dict[AgentRole, Dict]] = None,
        ema_alpha: float = 0.1,
        min_importance_threshold: float = 0.3,
    ):
        """
        Initialize DyLAN network.

        Args:
            agent_config: Custom agent configuration (uses DEFAULT_AGENT_CONFIG if None)
            ema_alpha: EMA decay rate for importance score updates
            min_importance_threshold: Minimum importance to avoid permanent skipping
        """
        self.agent_config = agent_config or DEFAULT_AGENT_CONFIG
        self.ema_alpha = ema_alpha
        self.min_importance_threshold = min_importance_threshold

        # Initialize agent stats
        self.agent_stats: Dict[AgentRole, AgentStats] = {
            role: AgentStats(agent_role=role, ema_alpha=ema_alpha)
            for role in AgentRole
        }

        # History for analysis
        self.contribution_history: List[AgentContribution] = []
        self.complexity_history: List[QueryComplexityResult] = []

        # Complexity indicators
        self._simple_patterns = [
            r"^what is\b",
            r"^define\b",
            r"^how to\b",
            r"^who is\b",
            r"^when was\b",
            r"\b[A-Z]{3,5}-?\d{2,4}\b",  # Error codes (SRVO-063, MOTN-023)
            r"\balarm\b.*\bmeaning\b",
            r"\bmeaning\b.*\balarm\b",
        ]
        self._complex_patterns = [
            r"\bcompare\b.*\bwith\b",
            r"\bcompare\b.*\band\b",  # "Compare X and Y"
            r"\bcompare\b.*\bexplain\b",  # "Compare...and explain"
            r"\band\b.*\band\b",
            r"\brelationship\b",
            r"\brelationship between\b",  # "relationship between X and Y"
            r"\bwhy\b.*\bhow\b",
            r"\bhistory\b.*\bevolution\b",
            r"\bmultiple\b.*\bsources\b",
            r"\binvestigate\b",
            r"\bdifferences?\b.*\bbetween\b",  # "differences between"
            r"\banalyze\b.*\b(and|with)\b",  # "analyze X and Y"
        ]

        self._exploratory_patterns = [
            r"\bexplore\b",
            r"\bresearch\b",
            r"\binvestigate\b.*\bhistory\b",
            r"\btrends?\b.*\bin\b",  # "trends in X"
            r"\bevolution\b.*\bof\b",  # "evolution of X"
            r"\bfuture\b.*\bof\b",  # "future of X"
            r"\bstate\s+of\s+(the\s+)?art\b",  # "state of the art"
            r"\boverview\b.*\bof\b",  # "overview of X"
        ]

        logger.info(f"DyLAN network initialized with {len(self.agent_config)} agents")

    async def classify_complexity(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> QueryComplexityResult:
        """
        Classify query complexity to determine agent routing.

        Args:
            query: User query string
            context: Optional context (conversation history, user preferences)

        Returns:
            QueryComplexityResult with complexity level and agent recommendations
        """
        import re

        query_lower = query.lower().strip()
        indicators = []
        confidence = 0.7  # Default confidence

        # Check simple patterns
        simple_matches = sum(
            1 for pattern in self._simple_patterns
            if re.search(pattern, query_lower, re.IGNORECASE)
        )
        if simple_matches > 0:
            indicators.append(f"simple_patterns_matched: {simple_matches}")

        # Check complex patterns
        complex_matches = sum(
            1 for pattern in self._complex_patterns
            if re.search(pattern, query_lower, re.IGNORECASE)
        )
        if complex_matches > 0:
            indicators.append(f"complex_patterns_matched: {complex_matches}")

        # Check exploratory patterns
        exploratory_matches = sum(
            1 for pattern in self._exploratory_patterns
            if re.search(pattern, query_lower, re.IGNORECASE)
        )
        if exploratory_matches > 0:
            indicators.append(f"exploratory_patterns_matched: {exploratory_matches}")

        # Word count heuristic
        word_count = len(query.split())
        if word_count <= 5:
            indicators.append(f"short_query: {word_count} words")
        elif word_count >= 15:
            indicators.append(f"long_query: {word_count} words")

        # Question word analysis
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        q_word_count = sum(1 for w in question_words if w in query_lower)
        if q_word_count > 1:
            indicators.append(f"multiple_question_words: {q_word_count}")

        # Context-based adjustments
        if context:
            if context.get("conversation_turns", 0) > 3:
                indicators.append("multi_turn_conversation")
            if context.get("user_expertise") == "expert":
                indicators.append("expert_user")

        # Determine complexity
        # Priority: check for exploratory indicators first, then complex, then simple

        # EXPLORATORY: Open-ended research queries
        if exploratory_matches >= 1 or \
           (word_count >= 20 and q_word_count >= 2) or \
           (complex_matches >= 2 and simple_matches == 0):
            complexity = QueryComplexity.EXPLORATORY
            confidence = 0.75
            skippable = []

        # COMPLEX: Multi-hop reasoning, comparisons, relationships
        elif complex_matches >= 1:
            complexity = QueryComplexity.COMPLEX
            confidence = 0.80
            skippable = [AgentRole.REFLECTOR]

        # SIMPLE: Direct lookups, definitions, short how-tos
        elif simple_matches >= 1 and complex_matches == 0 and word_count <= 10:
            complexity = QueryComplexity.SIMPLE
            confidence = 0.85
            skippable = [AgentRole.EVALUATOR, AgentRole.SCRAPER, AgentRole.VERIFIER, AgentRole.REFLECTOR]

        # MODERATE: Everything else
        else:
            complexity = QueryComplexity.MODERATE
            confidence = 0.75
            skippable = [AgentRole.EVALUATOR, AgentRole.REFLECTOR]

        # Determine recommended agents (all agents minus skippable)
        recommended = [role for role in AgentRole if role not in skippable]

        result = QueryComplexityResult(
            complexity=complexity,
            confidence=confidence,
            indicators=indicators,
            recommended_agents=recommended,
            skippable_agents=skippable,
        )

        self.complexity_history.append(result)
        if len(self.complexity_history) > 1000:
            self.complexity_history = self.complexity_history[-500:]

        logger.debug(f"Query complexity: {complexity.value} (confidence: {confidence:.2f})")
        return result

    def should_skip_agent(
        self,
        agent_role: AgentRole,
        complexity: QueryComplexityResult,
        current_confidence: float,
        quality_so_far: float = 0.5,
    ) -> SkipDecision:
        """
        Decide whether to skip an agent based on current state.

        Args:
            agent_role: Role of agent to evaluate
            complexity: Query complexity classification
            current_confidence: Current retrieval/synthesis confidence
            quality_so_far: Quality score from previous agents

        Returns:
            SkipDecision with recommendation
        """
        config = self.agent_config.get(agent_role, {})
        stats = self.agent_stats[agent_role]

        # Check if agent should always run
        if config.get("always_run", False):
            return SkipDecision(
                agent_role=agent_role,
                should_skip=False,
                reason="always_run_agent",
                confidence=1.0,
            )

        # Check if agent is in skippable list for this complexity
        if agent_role in complexity.skippable_agents:
            skip_threshold = config.get("skip_threshold", 0.85)

            if current_confidence >= skip_threshold:
                return SkipDecision(
                    agent_role=agent_role,
                    should_skip=True,
                    reason=f"confidence_above_threshold ({current_confidence:.2f} >= {skip_threshold})",
                    confidence=current_confidence,
                    alternative_action="use_cached_result",
                )

        # Check importance score - don't skip if agent has proven valuable
        if stats.importance_score >= 0.7 and not config.get("always_run"):
            return SkipDecision(
                agent_role=agent_role,
                should_skip=False,
                reason=f"high_importance_score ({stats.importance_score:.2f})",
                confidence=stats.importance_score,
            )

        # Check minimum importance threshold - avoid permanent skipping
        if stats.importance_score < self.min_importance_threshold:
            # Occasionally run to re-evaluate importance
            import random
            if random.random() < 0.2:  # 20% chance to run
                return SkipDecision(
                    agent_role=agent_role,
                    should_skip=False,
                    reason="exploration_run (low_importance_recheck)",
                    confidence=0.2,
                )

        # Default: don't skip unless explicitly in skippable list
        if agent_role not in complexity.skippable_agents:
            return SkipDecision(
                agent_role=agent_role,
                should_skip=False,
                reason="required_for_complexity",
                confidence=complexity.confidence,
            )

        return SkipDecision(
            agent_role=agent_role,
            should_skip=True,
            reason="skippable_for_complexity",
            confidence=complexity.confidence,
        )

    def get_active_agents(
        self,
        complexity: QueryComplexityResult,
        current_confidence: float = 0.0,
    ) -> List[AgentRole]:
        """
        Get list of agents that should run for this query.

        Args:
            complexity: Query complexity classification
            current_confidence: Current pipeline confidence

        Returns:
            List of AgentRole that should be executed
        """
        active = []
        skipped = []

        for role in AgentRole:
            decision = self.should_skip_agent(role, complexity, current_confidence)
            if decision.should_skip:
                skipped.append(role)
                logger.debug(f"Skipping {role.value}: {decision.reason}")
            else:
                active.append(role)

        logger.info(
            f"Active agents: {[a.value for a in active]}, "
            f"Skipped: {[s.value for s in skipped]}"
        )
        return active

    def record_contribution(self, contribution: AgentContribution) -> None:
        """
        Record an agent's contribution and update importance scores.

        Args:
            contribution: AgentContribution from completed agent execution
        """
        stats = self.agent_stats[contribution.agent_role]
        stats.update_importance(contribution)

        self.contribution_history.append(contribution)
        if len(self.contribution_history) > 10000:
            self.contribution_history = self.contribution_history[-5000:]

        logger.debug(
            f"Recorded {contribution.agent_role.value}: "
            f"quality_delta={contribution.quality_delta:.3f}, "
            f"new_importance={stats.importance_score:.3f}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate network statistics."""
        total_contributions = len(self.contribution_history)
        recent_contributions = self.contribution_history[-100:]

        # Calculate aggregate metrics
        total_runs = sum(s.total_runs for s in self.agent_stats.values())
        total_skips = sum(s.total_skips for s in self.agent_stats.values())
        total_time = sum(s.total_time_ms for s in self.agent_stats.values())

        # Complexity distribution
        complexity_dist = {}
        for c in self.complexity_history[-100:]:
            complexity_dist[c.complexity.value] = complexity_dist.get(c.complexity.value, 0) + 1

        return {
            "total_contributions": total_contributions,
            "total_runs": total_runs,
            "total_skips": total_skips,
            "overall_skip_rate": round(total_skips / max(total_runs + total_skips, 1), 4),
            "total_time_saved_ms": round(
                sum(
                    self.agent_stats[c.agent_role].avg_time_ms
                    for c in recent_contributions
                    if c.was_skipped
                ),
                2
            ),
            "complexity_distribution": complexity_dist,
            "agent_stats": {
                role.value: stats.to_dict()
                for role, stats in self.agent_stats.items()
            },
        }

    def reset_statistics(self) -> None:
        """Reset all statistics (for testing)."""
        self.agent_stats = {
            role: AgentStats(agent_role=role, ema_alpha=self.ema_alpha)
            for role in AgentRole
        }
        self.contribution_history.clear()
        self.complexity_history.clear()
        logger.info("DyLAN network statistics reset")


# Singleton instance
_dylan_network: Optional[DyLANAgentNetwork] = None


def get_dylan_network() -> DyLANAgentNetwork:
    """Get or create singleton DyLAN network instance."""
    global _dylan_network
    if _dylan_network is None:
        _dylan_network = DyLANAgentNetwork()
    return _dylan_network


def reset_dylan_network() -> None:
    """Reset singleton (for testing)."""
    global _dylan_network
    _dylan_network = None
