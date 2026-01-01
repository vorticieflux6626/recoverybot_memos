"""
Constraint Verification Gate for Agentic Search Pipeline

Validates output against active constraints before returning results.
Based on Eidoku/BEAVER patterns for output verification gates.

Part L.5 of Directive Propagation Enhancement (2026-01-01)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of constraints that can be enforced."""
    DOMAIN = "domain"           # Priority domain constraints (e.g., "fanuc", "robotics")
    TOPIC = "topic"             # Key topic constraints (e.g., "servo", "override")
    SAFETY = "safety"           # Safety-related constraints
    FORMAT = "format"           # Output format constraints
    SOURCE = "source"           # Source validation constraints
    CONTENT = "content"         # Content quality constraints


class ViolationSeverity(str, Enum):
    """Severity levels for constraint violations."""
    CRITICAL = "critical"       # Must fix before returning
    WARNING = "warning"         # Should note but can proceed
    INFO = "info"               # Informational only


@dataclass
class Constraint:
    """A single constraint to be verified."""
    constraint_type: ConstraintType
    value: str
    source: str = "analyzer"
    severity: ViolationSeverity = ViolationSeverity.WARNING
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        """Create Constraint from dictionary."""
        return cls(
            constraint_type=ConstraintType(data.get("type", "content")),
            value=data.get("value", ""),
            source=data.get("source", "analyzer"),
            severity=ViolationSeverity(data.get("severity", "warning")),
            description=data.get("description")
        )


@dataclass
class ConstraintViolation:
    """A constraint violation detected in the output."""
    constraint: Constraint
    reason: str
    location: Optional[str] = None  # Where in the output the violation occurred
    suggestion: Optional[str] = None  # How to fix the violation


@dataclass
class VerificationResult:
    """Result of constraint verification."""
    passed: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    checked_constraints: int = 0
    satisfied_constraints: int = 0
    verification_time_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def satisfaction_rate(self) -> float:
        """Calculate constraint satisfaction rate."""
        if self.checked_constraints == 0:
            return 1.0
        return self.satisfied_constraints / self.checked_constraints

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/API response."""
        return {
            "passed": self.passed,
            "violations": [
                {
                    "type": v.constraint.constraint_type.value,
                    "value": v.constraint.value,
                    "severity": v.constraint.severity.value,
                    "reason": v.reason,
                    "location": v.location,
                    "suggestion": v.suggestion
                }
                for v in self.violations
            ],
            "checked_constraints": self.checked_constraints,
            "satisfied_constraints": self.satisfied_constraints,
            "satisfaction_rate": self.satisfaction_rate,
            "verification_time_ms": self.verification_time_ms,
            "timestamp": self.timestamp
        }


class ConstraintVerificationGate:
    """
    Verification gate that validates output against active constraints.

    Based on Eidoku/BEAVER patterns for output verification:
    - Validates output before returning to user
    - Tracks constraint violations
    - Provides suggestions for fixing issues

    Usage:
        gate = ConstraintVerificationGate()
        result = await gate.verify(
            output=synthesis,
            constraints=state.active_constraints,
            sources=sources
        )
        if not result.passed:
            # Handle violations
            for v in result.violations:
                logger.warning(f"Constraint violated: {v.constraint.value}")
    """

    def __init__(self):
        """Initialize the verification gate."""
        self.total_verifications = 0
        self.total_violations = 0
        self.violations_by_type: Dict[str, int] = {}

    async def verify(
        self,
        output: str,
        constraints: List[Dict[str, Any]],
        sources: Optional[List[Dict[str, str]]] = None,
        key_topics: Optional[List[str]] = None,
        priority_domains: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Verify output against all active constraints.

        Args:
            output: The synthesized output to verify
            constraints: List of active constraint dictionaries
            sources: List of source dictionaries with URLs
            key_topics: Key topics that should be addressed
            priority_domains: Priority domains that sources should come from

        Returns:
            VerificationResult with pass/fail status and any violations
        """
        import time
        start_time = time.time()

        self.total_verifications += 1
        violations: List[ConstraintViolation] = []

        # Convert constraint dicts to Constraint objects
        constraint_objects = [Constraint.from_dict(c) for c in constraints]

        # Check each constraint
        for constraint in constraint_objects:
            violation = self._check_constraint(constraint, output, sources)
            if violation:
                violations.append(violation)
                self.total_violations += 1
                type_key = constraint.constraint_type.value
                self.violations_by_type[type_key] = self.violations_by_type.get(type_key, 0) + 1

        # Check key topics coverage if provided
        if key_topics:
            topic_violations = self._check_topic_coverage(output, key_topics)
            violations.extend(topic_violations)

        # Check priority domain sources if provided
        if priority_domains and sources:
            domain_violations = self._check_domain_sources(sources, priority_domains)
            violations.extend(domain_violations)

        # Determine if verification passed
        # Only critical violations cause failure
        critical_violations = [
            v for v in violations
            if v.constraint.severity == ViolationSeverity.CRITICAL
        ]
        passed = len(critical_violations) == 0

        verification_time_ms = int((time.time() - start_time) * 1000)

        result = VerificationResult(
            passed=passed,
            violations=violations,
            checked_constraints=len(constraint_objects) + len(key_topics or []) + (1 if priority_domains else 0),
            satisfied_constraints=len(constraint_objects) + len(key_topics or []) + (1 if priority_domains else 0) - len(violations),
            verification_time_ms=verification_time_ms
        )

        # Log verification result
        if not passed:
            logger.warning(
                f"Constraint verification FAILED: {len(critical_violations)} critical violations, "
                f"{len(violations)} total violations"
            )
        else:
            logger.info(
                f"Constraint verification passed: {result.satisfaction_rate:.1%} satisfaction rate, "
                f"{len(violations)} warnings"
            )

        return result

    def _check_constraint(
        self,
        constraint: Constraint,
        output: str,
        sources: Optional[List[Dict[str, str]]]
    ) -> Optional[ConstraintViolation]:
        """Check a single constraint against output."""
        output_lower = output.lower()
        value_lower = constraint.value.lower()

        if constraint.constraint_type == ConstraintType.DOMAIN:
            # Domain constraint: check if domain is mentioned or sources are from domain
            if value_lower not in output_lower:
                # Check if any source is from the domain
                source_match = False
                if sources:
                    for source in sources:
                        url = source.get("url", "").lower()
                        if value_lower in url:
                            source_match = True
                            break

                if not source_match:
                    return ConstraintViolation(
                        constraint=constraint,
                        reason=f"Domain '{constraint.value}' not addressed in output or sources",
                        suggestion=f"Ensure output addresses {constraint.value}-specific content"
                    )

        elif constraint.constraint_type == ConstraintType.TOPIC:
            # Topic constraint: check if topic is addressed
            if value_lower not in output_lower:
                return ConstraintViolation(
                    constraint=constraint,
                    reason=f"Key topic '{constraint.value}' not found in output",
                    suggestion=f"Include information about {constraint.value}"
                )

        elif constraint.constraint_type == ConstraintType.SAFETY:
            # Safety constraint: check for safety warnings if topic is sensitive
            safety_terms = ["warning", "caution", "safety", "danger", "hazard", "risk"]
            if not any(term in output_lower for term in safety_terms):
                return ConstraintViolation(
                    constraint=constraint,
                    reason="Safety-related content lacks safety warnings",
                    suggestion="Add appropriate safety warnings or precautions",
                    location="missing"
                )

        elif constraint.constraint_type == ConstraintType.SOURCE:
            # Source validation: check if sources meet criteria
            if sources:
                valid_sources = 0
                for source in sources:
                    url = source.get("url", "").lower()
                    if value_lower in url:
                        valid_sources += 1

                if valid_sources == 0:
                    return ConstraintViolation(
                        constraint=constraint,
                        reason=f"No sources from required domain: {constraint.value}",
                        suggestion=f"Include sources from {constraint.value}"
                    )

        elif constraint.constraint_type == ConstraintType.CONTENT:
            # Content quality: check if output contains required content
            if value_lower not in output_lower:
                return ConstraintViolation(
                    constraint=constraint,
                    reason=f"Required content '{constraint.value}' not found",
                    suggestion=f"Include information about {constraint.value}"
                )

        return None

    def _check_topic_coverage(
        self,
        output: str,
        key_topics: List[str]
    ) -> List[ConstraintViolation]:
        """Check if key topics are covered in output."""
        violations = []
        output_lower = output.lower()

        for topic in key_topics:
            if topic.lower() not in output_lower:
                violations.append(ConstraintViolation(
                    constraint=Constraint(
                        constraint_type=ConstraintType.TOPIC,
                        value=topic,
                        source="key_topics",
                        severity=ViolationSeverity.WARNING
                    ),
                    reason=f"Key topic '{topic}' not covered in output",
                    suggestion=f"Add information about {topic}"
                ))

        return violations

    def _check_domain_sources(
        self,
        sources: List[Dict[str, str]],
        priority_domains: List[str]
    ) -> List[ConstraintViolation]:
        """Check if sources come from priority domains."""
        violations = []

        # Count sources from priority domains
        priority_count = 0
        for source in sources:
            url = source.get("url", "").lower()
            for domain in priority_domains:
                if domain.lower() in url:
                    priority_count += 1
                    break

        # If less than 30% of sources are from priority domains, warn
        if len(sources) > 0:
            priority_ratio = priority_count / len(sources)
            if priority_ratio < 0.3:
                violations.append(ConstraintViolation(
                    constraint=Constraint(
                        constraint_type=ConstraintType.SOURCE,
                        value=", ".join(priority_domains[:3]),
                        source="priority_domains",
                        severity=ViolationSeverity.WARNING
                    ),
                    reason=f"Only {priority_ratio:.0%} of sources from priority domains",
                    suggestion=f"Include more sources from: {', '.join(priority_domains[:3])}"
                ))

        return violations

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return {
            "total_verifications": self.total_verifications,
            "total_violations": self.total_violations,
            "violations_by_type": self.violations_by_type,
            "average_violations_per_verification": (
                self.total_violations / self.total_verifications
                if self.total_verifications > 0 else 0
            )
        }


# Singleton instance
_constraint_gate: Optional[ConstraintVerificationGate] = None


def get_constraint_verification_gate() -> ConstraintVerificationGate:
    """Get or create the singleton constraint verification gate."""
    global _constraint_gate
    if _constraint_gate is None:
        _constraint_gate = ConstraintVerificationGate()
    return _constraint_gate
