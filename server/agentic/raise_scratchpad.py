"""
RAISE Four-Component Scratchpad Structure

Based on RAISE research (arXiv 2401.02777) for structured working memory.

Key insight: Agent working memory should be organized into four distinct
components that serve different purposes in the reasoning process:

1. Observations - Tool outputs, retrieved documents, external data
2. Reasoning - Intermediate conclusions, logical steps, confidence levels
3. Examples - Successful patterns from current session
4. Trajectory - Execution history with timestamps and outcomes

This structure enables:
- Better quality signal extraction from scratchpad contents
- Clear separation of facts vs reasoning
- Pattern reuse within session
- Debugging via trajectory analysis

References:
- RAISE: Retrieval-Augmented Instruction-Supervised Execution (arXiv 2401.02777)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)


class ObservationType(str, Enum):
    """Types of observations."""
    TOOL_OUTPUT = "tool_output"        # Result from a tool call
    DOCUMENT = "document"              # Retrieved document content
    SEARCH_RESULT = "search_result"    # Web search result
    USER_INPUT = "user_input"          # User-provided information
    SYSTEM_DATA = "system_data"        # System-provided data
    SCRAPED_CONTENT = "scraped_content"  # Scraped web content


class ReasoningType(str, Enum):
    """Types of reasoning steps."""
    DEDUCTION = "deduction"            # Logical deduction from facts
    INDUCTION = "induction"            # Generalization from examples
    ABDUCTION = "abduction"            # Best explanation inference
    COMPARISON = "comparison"          # Comparing alternatives
    SYNTHESIS = "synthesis"            # Combining information
    CRITIQUE = "critique"              # Evaluating claims
    HYPOTHESIS = "hypothesis"          # Tentative conclusion


class UncertaintyIndicator(str, Enum):
    """Indicators of uncertainty in reasoning."""
    HIGH_CONFIDENCE = "high_confidence"
    MEDIUM_CONFIDENCE = "medium_confidence"
    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    MISSING_INFORMATION = "missing_information"
    UNVERIFIED = "unverified"


@dataclass
class Observation:
    """An observation from tool output or external source."""
    id: str
    observation_type: ObservationType
    content: str
    source: str                        # URL or tool name
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.5         # 0-1 quality/relevance
    is_relevant: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "observation_type": self.observation_type.value,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "is_relevant": self.is_relevant
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            id=data["id"],
            observation_type=ObservationType(data["observation_type"]),
            content=data["content"],
            source=data["source"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            quality_score=data.get("quality_score", 0.5),
            is_relevant=data.get("is_relevant", True)
        )


@dataclass
class ReasoningStep:
    """An intermediate reasoning step."""
    id: str
    reasoning_type: ReasoningType
    conclusion: str
    premises: List[str]                # IDs of supporting observations/steps
    confidence: float = 0.5            # 0-1 confidence in conclusion
    uncertainty: UncertaintyIndicator = UncertaintyIndicator.MEDIUM_CONFIDENCE
    timestamp: float = field(default_factory=time.time)
    agent: str = ""                    # Which agent produced this
    notes: str = ""                    # Additional reasoning notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "reasoning_type": self.reasoning_type.value,
            "conclusion": self.conclusion,
            "premises": self.premises,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty.value,
            "timestamp": self.timestamp,
            "agent": self.agent,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        return cls(
            id=data["id"],
            reasoning_type=ReasoningType(data["reasoning_type"]),
            conclusion=data["conclusion"],
            premises=data.get("premises", []),
            confidence=data.get("confidence", 0.5),
            uncertainty=UncertaintyIndicator(data.get("uncertainty", "medium_confidence")),
            timestamp=data.get("timestamp", time.time()),
            agent=data.get("agent", ""),
            notes=data.get("notes", "")
        )


@dataclass
class Example:
    """A successful pattern from the current session."""
    id: str
    pattern_type: str                  # Type of pattern (e.g., "query_refinement")
    input_pattern: str                 # What triggered this pattern
    output_pattern: str                # What the pattern produced
    context: Dict[str, Any] = field(default_factory=dict)
    success_score: float = 0.8         # How successful this pattern was
    timestamp: float = field(default_factory=time.time)
    reuse_count: int = 0               # How many times reused in session

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "input_pattern": self.input_pattern,
            "output_pattern": self.output_pattern,
            "context": self.context,
            "success_score": self.success_score,
            "timestamp": self.timestamp,
            "reuse_count": self.reuse_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Example":
        return cls(
            id=data["id"],
            pattern_type=data["pattern_type"],
            input_pattern=data["input_pattern"],
            output_pattern=data["output_pattern"],
            context=data.get("context", {}),
            success_score=data.get("success_score", 0.8),
            timestamp=data.get("timestamp", time.time()),
            reuse_count=data.get("reuse_count", 0)
        )


@dataclass
class TrajectoryStep:
    """A step in the execution trajectory."""
    id: str
    action: str                        # What action was taken
    agent: str                         # Which agent took the action
    input_state: Dict[str, Any]        # State before action
    output_state: Dict[str, Any]       # State after action
    duration_ms: float = 0.0           # How long the action took
    timestamp: float = field(default_factory=time.time)
    success: bool = True               # Whether action succeeded
    error: Optional[str] = None        # Error message if failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "agent": self.agent,
            "input_state": self.input_state,
            "output_state": self.output_state,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        return cls(
            id=data["id"],
            action=data["action"],
            agent=data["agent"],
            input_state=data.get("input_state", {}),
            output_state=data.get("output_state", {}),
            duration_ms=data.get("duration_ms", 0.0),
            timestamp=data.get("timestamp", time.time()),
            success=data.get("success", True),
            error=data.get("error")
        )


@dataclass
class QualitySignal:
    """Quality signal extracted from scratchpad contents."""
    overall_quality: float             # 0-1 overall quality estimate
    evidence_quality: float            # Quality of observations
    reasoning_quality: float           # Quality of reasoning chain
    coverage: float                    # How much of query is answered
    uncertainty_indicators: List[str]  # List of uncertainty reasons
    recommendations: List[str]         # Suggested next actions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_quality": self.overall_quality,
            "evidence_quality": self.evidence_quality,
            "reasoning_quality": self.reasoning_quality,
            "coverage": self.coverage,
            "uncertainty_indicators": self.uncertainty_indicators,
            "recommendations": self.recommendations
        }


class RAISEScratchpad:
    """
    Four-component working memory structure based on RAISE.

    Components:
    1. observations: Tool outputs, retrieved docs
    2. reasoning: Intermediate conclusions, confidence
    3. examples: Successful patterns from session
    4. trajectory: Execution history with timestamps
    """

    def __init__(self, request_id: str, query: str):
        self.request_id = request_id
        self.query = query
        self.created_at = time.time()

        # Four RAISE components
        self.observations: Dict[str, Observation] = {}
        self.reasoning: Dict[str, ReasoningStep] = {}
        self.examples: Dict[str, Example] = {}
        self.trajectory: List[TrajectoryStep] = []

        # Cross-references
        self._observation_index: Dict[str, List[str]] = {}  # source -> [obs_ids]
        self._reasoning_chain: List[str] = []  # Ordered reasoning steps

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        hash_input = f"{prefix}:{self.request_id}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    # ========================================
    # OBSERVATIONS
    # ========================================

    def add_observation(
        self,
        content: str,
        observation_type: ObservationType,
        source: str,
        quality_score: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add an observation from a tool or external source."""
        obs_id = self._generate_id("obs")

        obs = Observation(
            id=obs_id,
            observation_type=observation_type,
            content=content,
            source=source,
            quality_score=quality_score,
            metadata=metadata or {}
        )

        self.observations[obs_id] = obs

        # Index by source
        if source not in self._observation_index:
            self._observation_index[source] = []
        self._observation_index[source].append(obs_id)

        logger.debug(f"Added observation {obs_id} from {source}")
        return obs_id

    def get_observations_by_type(
        self,
        observation_type: ObservationType
    ) -> List[Observation]:
        """Get all observations of a specific type."""
        return [
            obs for obs in self.observations.values()
            if obs.observation_type == observation_type
        ]

    def get_relevant_observations(
        self,
        min_quality: float = 0.5
    ) -> List[Observation]:
        """Get all relevant, high-quality observations."""
        return [
            obs for obs in self.observations.values()
            if obs.is_relevant and obs.quality_score >= min_quality
        ]

    # ========================================
    # REASONING
    # ========================================

    def add_reasoning_step(
        self,
        conclusion: str,
        reasoning_type: ReasoningType,
        premises: List[str] = None,
        confidence: float = 0.5,
        agent: str = "",
        notes: str = ""
    ) -> str:
        """Add an intermediate reasoning step."""
        step_id = self._generate_id("rsn")

        # Determine uncertainty indicator
        if confidence >= 0.8:
            uncertainty = UncertaintyIndicator.HIGH_CONFIDENCE
        elif confidence >= 0.5:
            uncertainty = UncertaintyIndicator.MEDIUM_CONFIDENCE
        else:
            uncertainty = UncertaintyIndicator.LOW_CONFIDENCE

        step = ReasoningStep(
            id=step_id,
            reasoning_type=reasoning_type,
            conclusion=conclusion,
            premises=premises or [],
            confidence=confidence,
            uncertainty=uncertainty,
            agent=agent,
            notes=notes
        )

        self.reasoning[step_id] = step
        self._reasoning_chain.append(step_id)

        logger.debug(
            f"Added reasoning step {step_id}: {reasoning_type.value} "
            f"(conf={confidence:.2f})"
        )
        return step_id

    def get_reasoning_chain(self) -> List[ReasoningStep]:
        """Get reasoning steps in order."""
        return [
            self.reasoning[sid]
            for sid in self._reasoning_chain
            if sid in self.reasoning
        ]

    def get_low_confidence_reasoning(
        self,
        threshold: float = 0.5
    ) -> List[ReasoningStep]:
        """Get reasoning steps below confidence threshold."""
        return [
            step for step in self.reasoning.values()
            if step.confidence < threshold
        ]

    # ========================================
    # EXAMPLES
    # ========================================

    def add_example(
        self,
        pattern_type: str,
        input_pattern: str,
        output_pattern: str,
        success_score: float = 0.8,
        context: Dict[str, Any] = None
    ) -> str:
        """Record a successful pattern for potential reuse."""
        example_id = self._generate_id("ex")

        example = Example(
            id=example_id,
            pattern_type=pattern_type,
            input_pattern=input_pattern,
            output_pattern=output_pattern,
            success_score=success_score,
            context=context or {}
        )

        self.examples[example_id] = example

        logger.debug(f"Added example {example_id}: {pattern_type}")
        return example_id

    def find_matching_example(
        self,
        pattern_type: str,
        input_pattern: str,
        similarity_threshold: float = 0.7
    ) -> Optional[Example]:
        """
        Find a previously successful example matching the pattern.

        Uses simple string matching for now; could be enhanced with embeddings.
        """
        best_match = None
        best_score = 0.0

        for example in self.examples.values():
            if example.pattern_type != pattern_type:
                continue

            # Simple overlap score
            input_words = set(input_pattern.lower().split())
            example_words = set(example.input_pattern.lower().split())

            if not input_words or not example_words:
                continue

            overlap = len(input_words.intersection(example_words))
            score = overlap / max(len(input_words), len(example_words))

            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = example

        if best_match:
            best_match.reuse_count += 1
            logger.debug(f"Found matching example {best_match.id} (reuse={best_match.reuse_count})")

        return best_match

    # ========================================
    # TRAJECTORY
    # ========================================

    def add_trajectory_step(
        self,
        action: str,
        agent: str,
        input_state: Dict[str, Any] = None,
        output_state: Dict[str, Any] = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None
    ) -> str:
        """Record an execution step in the trajectory."""
        step_id = self._generate_id("traj")

        step = TrajectoryStep(
            id=step_id,
            action=action,
            agent=agent,
            input_state=input_state or {},
            output_state=output_state or {},
            duration_ms=duration_ms,
            success=success,
            error=error
        )

        self.trajectory.append(step)

        logger.debug(
            f"Added trajectory step {step_id}: {action} by {agent} "
            f"(success={success}, {duration_ms:.0f}ms)"
        )
        return step_id

    def get_failed_steps(self) -> List[TrajectoryStep]:
        """Get all failed trajectory steps."""
        return [step for step in self.trajectory if not step.success]

    def get_recent_trajectory(self, n: int = 5) -> List[TrajectoryStep]:
        """Get the N most recent trajectory steps."""
        return self.trajectory[-n:]

    # ========================================
    # QUALITY SIGNAL
    # ========================================

    def get_quality_signal(self) -> QualitySignal:
        """
        Extract quality signal from scratchpad contents.

        Analyzes all four components to determine:
        - Overall quality
        - Areas of uncertainty
        - Recommended next actions
        """
        uncertainty_indicators = []
        recommendations = []

        # Evidence quality (observations)
        if not self.observations:
            evidence_quality = 0.0
            uncertainty_indicators.append("No observations gathered")
            recommendations.append("Gather more evidence via search or tools")
        else:
            relevant_obs = self.get_relevant_observations()
            evidence_quality = (
                sum(obs.quality_score for obs in relevant_obs) / len(relevant_obs)
                if relevant_obs else 0.0
            )
            if evidence_quality < 0.5:
                uncertainty_indicators.append("Low quality observations")
                recommendations.append("Find higher quality sources")

        # Reasoning quality
        if not self.reasoning:
            reasoning_quality = 0.0
            uncertainty_indicators.append("No reasoning steps")
            recommendations.append("Develop reasoning chain from observations")
        else:
            chain = self.get_reasoning_chain()
            reasoning_quality = (
                sum(step.confidence for step in chain) / len(chain)
                if chain else 0.0
            )

            # Check for low-confidence steps
            low_conf = self.get_low_confidence_reasoning(0.5)
            if low_conf:
                uncertainty_indicators.append(
                    f"{len(low_conf)} low-confidence reasoning steps"
                )
                recommendations.append("Verify low-confidence conclusions")

            # Check for conflicting evidence
            conflicting = [
                step for step in chain
                if step.uncertainty == UncertaintyIndicator.CONFLICTING_EVIDENCE
            ]
            if conflicting:
                uncertainty_indicators.append("Conflicting evidence detected")
                recommendations.append("Resolve contradictions")

        # Coverage estimate (simplified)
        # In a real implementation, compare against decomposed questions
        coverage = min(1.0, len(self.observations) / 5)  # Simplified

        # Overall quality
        overall_quality = (
            0.4 * evidence_quality +
            0.4 * reasoning_quality +
            0.2 * coverage
        )

        # Additional recommendations based on trajectory
        if self.trajectory:
            failed = self.get_failed_steps()
            if failed:
                recommendations.append(
                    f"Retry {len(failed)} failed actions"
                )

        return QualitySignal(
            overall_quality=overall_quality,
            evidence_quality=evidence_quality,
            reasoning_quality=reasoning_quality,
            coverage=coverage,
            uncertainty_indicators=uncertainty_indicators,
            recommendations=recommendations
        )

    # ========================================
    # SERIALIZATION
    # ========================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the RAISE scratchpad."""
        return {
            "request_id": self.request_id,
            "query": self.query,
            "created_at": self.created_at,
            "observations": {
                oid: obs.to_dict() for oid, obs in self.observations.items()
            },
            "reasoning": {
                rid: step.to_dict() for rid, step in self.reasoning.items()
            },
            "examples": {
                eid: ex.to_dict() for eid, ex in self.examples.items()
            },
            "trajectory": [step.to_dict() for step in self.trajectory],
            "reasoning_chain": self._reasoning_chain
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAISEScratchpad":
        """Deserialize a RAISE scratchpad."""
        scratchpad = cls(
            request_id=data["request_id"],
            query=data["query"]
        )
        scratchpad.created_at = data.get("created_at", time.time())

        for obs_id, obs_data in data.get("observations", {}).items():
            scratchpad.observations[obs_id] = Observation.from_dict(obs_data)

        for step_id, step_data in data.get("reasoning", {}).items():
            scratchpad.reasoning[step_id] = ReasoningStep.from_dict(step_data)

        for ex_id, ex_data in data.get("examples", {}).items():
            scratchpad.examples[ex_id] = Example.from_dict(ex_data)

        for step_data in data.get("trajectory", []):
            scratchpad.trajectory.append(TrajectoryStep.from_dict(step_data))

        scratchpad._reasoning_chain = data.get("reasoning_chain", [])

        return scratchpad

    def get_summary(self) -> str:
        """Get a brief summary of scratchpad contents."""
        quality = self.get_quality_signal()
        return (
            f"RAISE Scratchpad Summary:\n"
            f"- Observations: {len(self.observations)}\n"
            f"- Reasoning steps: {len(self.reasoning)}\n"
            f"- Examples: {len(self.examples)}\n"
            f"- Trajectory steps: {len(self.trajectory)}\n"
            f"- Overall quality: {quality.overall_quality:.2f}\n"
            f"- Uncertainties: {', '.join(quality.uncertainty_indicators) or 'None'}"
        )


# Factory function
def create_raise_scratchpad(request_id: str, query: str) -> RAISEScratchpad:
    """Create a new RAISE scratchpad instance."""
    return RAISEScratchpad(request_id=request_id, query=query)
