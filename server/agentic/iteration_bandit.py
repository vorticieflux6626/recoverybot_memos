"""
UCB Bandit for Iteration Decisions Module

Based on REFRAIN (arXiv 2510.10103) research achieving 20-55% token reduction.

Key insight: Use multi-armed bandit algorithms to balance exploration (trying new
approaches) with exploitation (using known effective strategies). This replaces
fixed iteration policies with adaptive decision-making.

UCB (Upper Confidence Bound) formula:
UCB(a) = Q(a) + c * sqrt(log(t) / N(a))

Where:
- Q(a) = average reward for action a
- N(a) = times action a was selected
- c = exploration constant (typically 2.0)
- t = total time steps

References:
- REFRAIN: arXiv 2510.10103 - Retrieval-Augmented Framework with Active INquiry
- UCB1: Auer et al., 2002 - Finite-time Analysis of MAB
- Thompson Sampling: Alternative Bayesian approach
"""

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from collections import defaultdict
import json
import time

logger = logging.getLogger(__name__)


class RefinementAction(str, Enum):
    """Available actions for the bandit to choose from."""
    SEARCH_MORE = "search_more"       # Conduct additional searches
    REFINE_QUERY = "refine_query"     # Rephrase/refine the search query
    SYNTHESIZE_NOW = "synthesize_now" # Stop iterating and synthesize
    DECOMPOSE = "decompose"           # Break query into sub-questions
    VERIFY_CLAIMS = "verify_claims"   # Verify existing claims before continuing
    BROADEN_SCOPE = "broaden_scope"   # Expand search to related topics
    NARROW_FOCUS = "narrow_focus"     # Focus on most relevant subset


@dataclass
class ActionOutcome:
    """Record of an action and its outcome."""
    action: RefinementAction
    reward: float  # 0-1 normalized reward
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ArmStats:
    """Statistics for a bandit arm (action)."""
    action: RefinementAction
    total_reward: float = 0.0
    pull_count: int = 0
    successes: int = 0
    failures: int = 0
    last_pulled: float = 0.0
    reward_history: List[float] = field(default_factory=list)

    @property
    def average_reward(self) -> float:
        """Q(a) - average reward for this arm."""
        if self.pull_count == 0:
            return 0.5  # Optimistic initialization
        return self.total_reward / self.pull_count

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5


@dataclass
class RefinementState:
    """Current state for bandit decision-making."""
    query: str
    iteration: int
    max_iterations: int
    current_confidence: float
    num_sources: int
    coverage_score: float
    entropy: float
    has_contradictions: bool
    unanswered_questions: List[str]
    recent_actions: List[RefinementAction] = field(default_factory=list)


@dataclass
class BanditDecision:
    """Decision made by the bandit."""
    action: RefinementAction
    confidence: float  # How confident the bandit is in this choice
    ucb_score: float
    reasoning: str
    alternative_actions: List[Tuple[RefinementAction, float]]  # (action, score)


class IterationBandit:
    """
    Multi-armed bandit for exploration/exploitation tradeoff in refinement.

    Instead of fixed rules like "if confidence < X, search more",
    the bandit learns from outcomes which actions work best in which contexts.

    REFRAIN research shows 20-55% token reduction through intelligent
    action selection based on accumulated experience.
    """

    def __init__(
        self,
        exploration_constant: float = 2.0,
        decay_factor: float = 0.95,  # Decay old rewards
        context_features: bool = True,
        persistence_path: Optional[str] = None
    ):
        self.exploration_constant = exploration_constant
        self.decay_factor = decay_factor
        self.context_features = context_features
        self.persistence_path = persistence_path

        # Initialize arms with optimistic values
        self.arms: Dict[RefinementAction, ArmStats] = {
            action: ArmStats(action=action)
            for action in RefinementAction
        }

        self.total_pulls = 0
        self.history: List[ActionOutcome] = []

        # Context-aware statistics (action performance by context)
        self.context_stats: Dict[str, Dict[RefinementAction, ArmStats]] = defaultdict(
            lambda: {a: ArmStats(action=a) for a in RefinementAction}
        )

        # Load persisted stats if available
        if persistence_path:
            self._load_stats()

    def _get_context_key(self, state: RefinementState) -> str:
        """
        Generate a context key for context-aware learning.

        Groups similar states together for more targeted learning.
        """
        conf_bucket = "high" if state.current_confidence > 0.7 else (
            "medium" if state.current_confidence > 0.4 else "low"
        )
        entropy_bucket = "high" if state.entropy > 0.5 else (
            "medium" if state.entropy > 0.2 else "low"
        )
        coverage_bucket = "good" if state.coverage_score > 0.7 else (
            "partial" if state.coverage_score > 0.4 else "low"
        )
        iter_bucket = "early" if state.iteration <= 2 else (
            "mid" if state.iteration <= 5 else "late"
        )

        return f"{conf_bucket}_{entropy_bucket}_{coverage_bucket}_{iter_bucket}"

    def _calculate_ucb(
        self,
        arm: ArmStats,
        total_pulls: int,
        context_arm: Optional[ArmStats] = None
    ) -> float:
        """
        Calculate Upper Confidence Bound for an arm.

        UCB(a) = Q(a) + c * sqrt(log(t) / N(a))

        If context_arm is provided, blend global and context-specific stats.
        """
        if arm.pull_count == 0:
            return float('inf')  # Never pulled = maximum exploration bonus

        # Average reward
        q_value = arm.average_reward

        # Blend with context-specific stats if available
        if context_arm and context_arm.pull_count > 0:
            context_weight = min(0.6, context_arm.pull_count / 10)
            q_value = (
                context_weight * context_arm.average_reward +
                (1 - context_weight) * arm.average_reward
            )

        # Exploration bonus
        exploration_bonus = self.exploration_constant * math.sqrt(
            math.log(total_pulls + 1) / arm.pull_count
        )

        return q_value + exploration_bonus

    def _apply_state_heuristics(
        self,
        state: RefinementState,
        ucb_scores: Dict[RefinementAction, float]
    ) -> Dict[RefinementAction, float]:
        """
        Apply domain-specific heuristics to adjust UCB scores.

        These encode domain knowledge while still allowing the bandit to learn.
        """
        adjusted = dict(ucb_scores)

        # If close to max iterations, boost SYNTHESIZE_NOW
        if state.iteration >= state.max_iterations - 1:
            adjusted[RefinementAction.SYNTHESIZE_NOW] *= 1.5

        # If confidence is very high, boost SYNTHESIZE_NOW
        if state.current_confidence > 0.85:
            adjusted[RefinementAction.SYNTHESIZE_NOW] *= 1.3
            adjusted[RefinementAction.SEARCH_MORE] *= 0.7

        # If entropy is high (uncertain), boost exploration actions
        if state.entropy > 0.5:
            adjusted[RefinementAction.SEARCH_MORE] *= 1.2
            adjusted[RefinementAction.DECOMPOSE] *= 1.2
            adjusted[RefinementAction.SYNTHESIZE_NOW] *= 0.8

        # If coverage is low, boost content-gathering actions
        if state.coverage_score < 0.5:
            adjusted[RefinementAction.SEARCH_MORE] *= 1.2
            adjusted[RefinementAction.BROADEN_SCOPE] *= 1.1

        # If there are contradictions, boost verification
        if state.has_contradictions:
            adjusted[RefinementAction.VERIFY_CLAIMS] *= 1.4

        # If many unanswered questions, boost decomposition
        if len(state.unanswered_questions) > 2:
            adjusted[RefinementAction.DECOMPOSE] *= 1.3

        # Penalize repeating recent actions (encourage diversity)
        for recent in state.recent_actions[-3:]:
            if recent in adjusted:
                adjusted[recent] *= 0.8

        # If few sources, boost search
        if state.num_sources < 3:
            adjusted[RefinementAction.SEARCH_MORE] *= 1.3

        return adjusted

    def select_action(
        self,
        state: RefinementState,
        exclude_actions: Optional[List[RefinementAction]] = None
    ) -> BanditDecision:
        """
        Select the best action using UCB with context-aware adjustments.

        Args:
            state: Current refinement state
            exclude_actions: Actions to exclude from consideration

        Returns:
            BanditDecision with selected action and reasoning
        """
        exclude = set(exclude_actions or [])

        # Get context-specific stats
        context_key = self._get_context_key(state) if self.context_features else None
        context_arms = self.context_stats.get(context_key, {}) if context_key else {}

        # Calculate UCB scores for all actions
        ucb_scores = {}
        for action, arm in self.arms.items():
            if action in exclude:
                continue

            context_arm = context_arms.get(action)
            ucb_scores[action] = self._calculate_ucb(
                arm, self.total_pulls, context_arm
            )

        # Apply state-based heuristics
        adjusted_scores = self._apply_state_heuristics(state, ucb_scores)

        # Select action with highest adjusted UCB
        if not adjusted_scores:
            # All actions excluded, fall back to synthesize
            return BanditDecision(
                action=RefinementAction.SYNTHESIZE_NOW,
                confidence=0.5,
                ucb_score=0.0,
                reasoning="All other actions excluded",
                alternative_actions=[]
            )

        # Sort by adjusted score
        sorted_actions = sorted(
            adjusted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_action, best_score = sorted_actions[0]

        # Calculate confidence in the decision
        if len(sorted_actions) > 1:
            second_best_score = sorted_actions[1][1]
            margin = best_score - second_best_score
            confidence = min(1.0, 0.5 + margin / 2)
        else:
            confidence = 0.9

        # Build reasoning
        arm = self.arms[best_action]
        context_arm = context_arms.get(best_action)
        context_info = (
            f", context success rate: {context_arm.success_rate:.1%}"
            if context_arm and context_arm.pull_count > 0
            else ""
        )

        reasoning = (
            f"Selected {best_action.value}: UCB={best_score:.3f}, "
            f"avg_reward={arm.average_reward:.2f}, "
            f"pulls={arm.pull_count}{context_info}"
        )

        # Alternative actions
        alternatives = [(action, score) for action, score in sorted_actions[1:4]]

        logger.debug(f"Bandit decision: {reasoning}")

        return BanditDecision(
            action=best_action,
            confidence=confidence,
            ucb_score=best_score,
            reasoning=reasoning,
            alternative_actions=alternatives
        )

    def record_outcome(
        self,
        action: RefinementAction,
        reward: float,
        state: Optional[RefinementState] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record the outcome of an action for learning.

        Args:
            action: The action that was taken
            reward: Normalized reward (0-1)
            state: The state when action was taken (for context learning)
            context: Additional context information
        """
        reward = max(0.0, min(1.0, reward))  # Clamp to [0, 1]

        # Update global arm statistics
        arm = self.arms[action]
        arm.total_reward += reward
        arm.pull_count += 1
        arm.last_pulled = time.time()
        arm.reward_history.append(reward)

        # Track successes/failures
        if reward >= 0.6:
            arm.successes += 1
        elif reward <= 0.4:
            arm.failures += 1

        # Keep only recent history
        if len(arm.reward_history) > 100:
            arm.reward_history = arm.reward_history[-100:]

        self.total_pulls += 1

        # Update context-specific statistics
        if state and self.context_features:
            context_key = self._get_context_key(state)
            context_arm = self.context_stats[context_key][action]
            context_arm.total_reward += reward
            context_arm.pull_count += 1
            context_arm.last_pulled = time.time()
            if reward >= 0.6:
                context_arm.successes += 1
            elif reward <= 0.4:
                context_arm.failures += 1

        # Record in history
        outcome = ActionOutcome(
            action=action,
            reward=reward,
            context=context or {}
        )
        self.history.append(outcome)

        # Keep only recent history
        if len(self.history) > 500:
            self.history = self.history[-500:]

        logger.debug(
            f"Recorded outcome: {action.value} -> reward={reward:.2f} "
            f"(new avg={arm.average_reward:.3f})"
        )

    def calculate_reward(
        self,
        action: RefinementAction,
        before_state: RefinementState,
        after_state: RefinementState,
        synthesis_quality: Optional[float] = None
    ) -> float:
        """
        Calculate reward based on state improvement.

        Reward is based on:
        - Confidence improvement
        - Coverage improvement
        - Entropy reduction
        - Synthesis quality (if applicable)
        """
        rewards = []

        # Confidence improvement
        conf_delta = after_state.current_confidence - before_state.current_confidence
        conf_reward = 0.5 + conf_delta  # Delta of 0.5 = reward of 1.0
        rewards.append(("confidence", conf_reward * 0.35))

        # Coverage improvement
        cov_delta = after_state.coverage_score - before_state.coverage_score
        cov_reward = 0.5 + cov_delta
        rewards.append(("coverage", cov_reward * 0.25))

        # Entropy reduction (lower is better)
        entropy_delta = before_state.entropy - after_state.entropy  # Positive = improvement
        entropy_reward = 0.5 + entropy_delta
        rewards.append(("entropy", entropy_reward * 0.20))

        # Synthesis quality (if action was SYNTHESIZE_NOW)
        if action == RefinementAction.SYNTHESIZE_NOW and synthesis_quality is not None:
            rewards.append(("synthesis", synthesis_quality * 0.20))

        # Action-specific bonuses
        if action == RefinementAction.SEARCH_MORE:
            # Reward for finding new sources
            source_delta = after_state.num_sources - before_state.num_sources
            if source_delta > 0:
                rewards.append(("new_sources", min(0.3, source_delta * 0.1)))

        if action == RefinementAction.DECOMPOSE:
            # Reward for resolving unanswered questions
            resolved = (
                len(before_state.unanswered_questions) -
                len(after_state.unanswered_questions)
            )
            if resolved > 0:
                rewards.append(("resolved_questions", min(0.3, resolved * 0.1)))

        # Calculate total reward
        total_reward = sum(r for _, r in rewards)
        total_reward = max(0.0, min(1.0, total_reward))

        logger.debug(
            f"Reward calculation for {action.value}: {total_reward:.3f} "
            f"({', '.join(f'{n}={r:.2f}' for n, r in rewards)})"
        )

        return total_reward

    def get_statistics(self) -> Dict[str, Any]:
        """Get bandit statistics for debugging/monitoring."""
        stats = {
            "total_pulls": self.total_pulls,
            "history_length": len(self.history),
            "arms": {}
        }

        for action, arm in self.arms.items():
            stats["arms"][action.value] = {
                "pulls": arm.pull_count,
                "avg_reward": round(arm.average_reward, 3),
                "success_rate": round(arm.success_rate, 3),
                "successes": arm.successes,
                "failures": arm.failures,
                "ucb_score": round(
                    self._calculate_ucb(arm, self.total_pulls),
                    3
                ) if arm.pull_count > 0 else "inf"
            }

        # Sort by UCB score
        stats["action_ranking"] = sorted(
            stats["arms"].keys(),
            key=lambda a: stats["arms"][a]["ucb_score"]
            if stats["arms"][a]["ucb_score"] != "inf" else float('inf'),
            reverse=True
        )

        return stats

    def _load_stats(self) -> bool:
        """Load persisted statistics."""
        if not self.persistence_path:
            return False

        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)

            for action_name, arm_data in data.get("arms", {}).items():
                try:
                    action = RefinementAction(action_name)
                    arm = self.arms[action]
                    arm.total_reward = arm_data.get("total_reward", 0)
                    arm.pull_count = arm_data.get("pull_count", 0)
                    arm.successes = arm_data.get("successes", 0)
                    arm.failures = arm_data.get("failures", 0)
                except (ValueError, KeyError):
                    continue

            self.total_pulls = data.get("total_pulls", 0)
            logger.info(f"Loaded bandit stats from {self.persistence_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load bandit stats: {e}")
            return False

    def save_stats(self) -> bool:
        """Persist statistics to disk."""
        if not self.persistence_path:
            return False

        try:
            data = {
                "total_pulls": self.total_pulls,
                "arms": {
                    action.value: {
                        "total_reward": arm.total_reward,
                        "pull_count": arm.pull_count,
                        "successes": arm.successes,
                        "failures": arm.failures
                    }
                    for action, arm in self.arms.items()
                }
            }

            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved bandit stats to {self.persistence_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save bandit stats: {e}")
            return False

    def reset(self) -> None:
        """Reset all statistics."""
        for arm in self.arms.values():
            arm.total_reward = 0.0
            arm.pull_count = 0
            arm.successes = 0
            arm.failures = 0
            arm.reward_history = []

        self.total_pulls = 0
        self.history = []
        self.context_stats.clear()

        logger.info("Bandit statistics reset")


# Singleton instance
_iteration_bandit: Optional[IterationBandit] = None


def get_iteration_bandit(
    persistence_path: Optional[str] = None
) -> IterationBandit:
    """Get or create the iteration bandit singleton."""
    global _iteration_bandit
    if _iteration_bandit is None:
        _iteration_bandit = IterationBandit(persistence_path=persistence_path)
    return _iteration_bandit
