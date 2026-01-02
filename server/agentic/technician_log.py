"""
Technician Log - Human-readable diagnostic summaries for robotics technicians.

Part of P1 Observability Enhancement (OBSERVABILITY_IMPROVEMENT_PLAN.md).

Generates technician-friendly log entries that explain:
- What the AI found and how confident it is
- The reasoning chain in human-readable steps
- Sources consulted with relevance scores
- What the AI is uncertain about
- Safety warnings and recommended actions
- Refinements made during the search

Based on industrial AI explainability research - technicians need
layered explanations with confidence visualization, not ML metrics.

Created: 2026-01-02
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SourceInfo:
    """Information about a consulted source."""
    title: str
    url: Optional[str] = None
    source_type: str = "web"  # web, documentation, manual, forum, video
    relevance: float = 0.0  # 0-1 relevance score
    trust_level: str = "unknown"  # official, trusted, community, unknown
    excerpt: Optional[str] = None  # Brief excerpt used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "type": self.source_type,
            "relevance": round(self.relevance, 2),
            "trust": self.trust_level,
            "excerpt": self.excerpt[:100] if self.excerpt else None
        }


@dataclass
class RefinementRecord:
    """Record of a refinement/correction made during search."""
    reason: str
    action: str
    iteration: int
    improvement: Optional[str] = None


@dataclass
class TechnicianLog:
    """
    Log entry optimized for technician consumption.

    Designed for robotics technicians who need:
    - Clear confidence visualization
    - Human-readable reasoning steps
    - Actionable recommendations
    - Safety warnings
    - Source attribution
    """

    # Header
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    query_summary: str = ""  # 1-line summary of what was asked

    # Confidence visualization
    confidence_score: float = 0.0
    confidence_factors: List[Tuple[str, float]] = field(default_factory=list)
    # e.g., [("source_quality", 0.9), ("verification", 0.8), ...]

    # Reasoning chain (human-readable)
    reasoning_steps: List[str] = field(default_factory=list)
    # e.g., ["Identified SRVO-063 as encoder position alarm", ...]

    # Source attribution
    sources_consulted: List[SourceInfo] = field(default_factory=list)

    # Uncertainty declaration
    uncertain_about: List[str] = field(default_factory=list)
    # e.g., ["Model-specific variations", "Controller version differences"]
    additional_info_needed: List[str] = field(default_factory=list)
    # e.g., ["Robot model", "Controller version", "When error occurs"]

    # Actionable output
    recommended_action: str = ""
    safety_warnings: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    # Backtracking history
    refinements_made: List[RefinementRecord] = field(default_factory=list)

    # Diagnostic context
    error_codes_identified: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)
    equipment_mentioned: List[str] = field(default_factory=list)

    @staticmethod
    def confidence_to_bar(score: float) -> str:
        """Convert 0-1 score to visual bar with emoji indicator."""
        filled = int(score * 10)
        empty = 10 - filled

        # Add confidence emoji
        if score >= 0.8:
            emoji = "✅"
        elif score >= 0.6:
            emoji = "🔶"
        elif score >= 0.4:
            emoji = "⚠️"
        else:
            emoji = "❓"

        return f"{emoji} {'█' * filled}{'░' * empty} {score:.0%}"

    @staticmethod
    def confidence_to_label(score: float) -> str:
        """Convert confidence score to human-readable label."""
        if score >= 0.9:
            return "Very High Confidence"
        elif score >= 0.8:
            return "High Confidence"
        elif score >= 0.6:
            return "Moderate Confidence"
        elif score >= 0.4:
            return "Low Confidence"
        else:
            return "Very Low Confidence - Verify Independently"

    def to_markdown(self) -> str:
        """Generate technician-friendly markdown summary."""
        lines = [
            f"# Diagnostic Summary",
            f"",
            f"**Query**: {self.query_summary}",
            f"",
            f"**Generated**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"---",
            f"",
            f"## Confidence: {self.confidence_to_bar(self.confidence_score)}",
            f"",
            f"*{self.confidence_to_label(self.confidence_score)}*",
            f"",
        ]

        # Confidence breakdown
        if self.confidence_factors:
            lines.append("### Confidence Factors:")
            for factor, score in self.confidence_factors:
                factor_bar = "█" * int(score * 5) + "░" * (5 - int(score * 5))
                lines.append(f"- **{factor}**: {factor_bar} {score:.0%}")
            lines.append("")

        # Error codes
        if self.error_codes_identified:
            lines.append("## Error Codes Identified")
            for code in self.error_codes_identified:
                lines.append(f"- `{code}`")
            lines.append("")

        # Reasoning chain
        if self.reasoning_steps:
            lines.append("## How I Reached This Conclusion")
            lines.append("")
            for i, step in enumerate(self.reasoning_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

        # Safety warnings (prominent)
        if self.safety_warnings:
            lines.append("## ⚠️ SAFETY WARNINGS")
            lines.append("")
            for warning in self.safety_warnings:
                lines.append(f"**⚠️ {warning}**")
            lines.append("")

        # Prerequisites
        if self.prerequisites:
            lines.append("## Prerequisites Before Proceeding")
            lines.append("")
            for prereq in self.prerequisites:
                lines.append(f"- [ ] {prereq}")
            lines.append("")

        # Recommended action
        if self.recommended_action:
            lines.append("## Recommended Next Step")
            lines.append("")
            lines.append(self.recommended_action)
            lines.append("")

        # Uncertainty
        if self.uncertain_about:
            lines.append("## What I'm Uncertain About")
            lines.append("")
            for item in self.uncertain_about:
                lines.append(f"- {item}")
            lines.append("")

        # Additional info needed
        if self.additional_info_needed:
            lines.append("## Additional Information Would Help")
            lines.append("")
            lines.append("*Providing these details could improve the diagnosis:*")
            for item in self.additional_info_needed:
                lines.append(f"- {item}")
            lines.append("")

        # Related errors
        if self.related_errors:
            lines.append("## Related Error Codes")
            lines.append("")
            lines.append("*These errors are often associated with this issue:*")
            for error in self.related_errors:
                lines.append(f"- `{error}`")
            lines.append("")

        # Sources
        if self.sources_consulted:
            lines.append("## Sources Consulted")
            lines.append("")
            for source in sorted(self.sources_consulted, key=lambda s: s.relevance, reverse=True)[:7]:
                rel_bar = "●" * int(source.relevance * 5) + "○" * (5 - int(source.relevance * 5))
                trust_badge = {
                    "official": "📗",
                    "trusted": "📘",
                    "community": "📙",
                    "unknown": "📓"
                }.get(source.trust_level, "📓")

                if source.url:
                    lines.append(f"- {trust_badge} [{source.title}]({source.url}) ({rel_bar} {source.relevance:.0%})")
                else:
                    lines.append(f"- {trust_badge} {source.title} ({rel_bar} {source.relevance:.0%})")
            lines.append("")

        # Refinements
        if self.refinements_made:
            lines.append("## Search Refinements Made")
            lines.append("")
            for ref in self.refinements_made:
                lines.append(f"- **Iteration {ref.iteration}**: {ref.reason} → {ref.action}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Request ID: {self.request_id}*")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate HTML version for web/app display."""
        # Convert markdown to basic HTML
        md = self.to_markdown()

        # Simple conversions
        html = md
        html = html.replace("# ", "<h1>").replace("\n\n<h1>", "</h1>\n<h1>")
        html = html.replace("## ", "<h2>").replace("\n\n<h2>", "</h2>\n<h2>")
        html = html.replace("### ", "<h3>").replace("\n\n<h3>", "</h3>\n<h3>")
        html = html.replace("**", "<strong>").replace("</strong><strong>", "")
        html = html.replace("*", "<em>").replace("</em><em>", "")
        html = html.replace("- ", "<li>").replace("\n<li>", "</li>\n<li>")
        html = html.replace("`", "<code>").replace("</code><code>", "")

        return f"<div class='technician-log'>{html}</div>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "query_summary": self.query_summary,
            "confidence": {
                "score": self.confidence_score,
                "label": self.confidence_to_label(self.confidence_score),
                "bar": self.confidence_to_bar(self.confidence_score),
                "factors": [{"name": n, "score": s} for n, s in self.confidence_factors]
            },
            "reasoning_steps": self.reasoning_steps,
            "sources": [s.to_dict() for s in self.sources_consulted],
            "uncertainty": {
                "items": self.uncertain_about,
                "info_needed": self.additional_info_needed
            },
            "action": {
                "recommended": self.recommended_action,
                "safety_warnings": self.safety_warnings,
                "prerequisites": self.prerequisites
            },
            "errors": {
                "identified": self.error_codes_identified,
                "related": self.related_errors
            },
            "equipment": self.equipment_mentioned,
            "refinements": [asdict(r) for r in self.refinements_made]
        }


class TechnicianLogBuilder:
    """
    Builder for constructing TechnicianLog entries from pipeline data.

    Usage:
        builder = TechnicianLogBuilder(request_id="req-123")
        builder.set_query("How to fix SRVO-063 alarm?")
        builder.set_confidence(0.85, [("sources", 0.9), ("verification", 0.8)])
        builder.add_reasoning_step("Identified SRVO-063 as encoder position alarm")
        builder.add_source("FANUC Manual", url="...", relevance=0.95)
        builder.add_safety_warning("Ensure robot is in emergency stop")
        builder.set_recommendation("Check encoder cable connections")

        log = builder.build()
        print(log.to_markdown())
    """

    def __init__(self, request_id: str):
        self.log = TechnicianLog(request_id=request_id)

    def set_query(self, query_summary: str) -> "TechnicianLogBuilder":
        """Set the query summary."""
        self.log.query_summary = query_summary
        return self

    def set_confidence(
        self,
        score: float,
        factors: Optional[List[Tuple[str, float]]] = None
    ) -> "TechnicianLogBuilder":
        """Set confidence score and factors."""
        self.log.confidence_score = score
        if factors:
            self.log.confidence_factors = factors
        return self

    def add_reasoning_step(self, step: str) -> "TechnicianLogBuilder":
        """Add a reasoning step."""
        self.log.reasoning_steps.append(step)
        return self

    def add_source(
        self,
        title: str,
        url: Optional[str] = None,
        relevance: float = 0.0,
        source_type: str = "web",
        trust_level: str = "unknown",
        excerpt: Optional[str] = None
    ) -> "TechnicianLogBuilder":
        """Add a consulted source."""
        self.log.sources_consulted.append(SourceInfo(
            title=title,
            url=url,
            source_type=source_type,
            relevance=relevance,
            trust_level=trust_level,
            excerpt=excerpt
        ))
        return self

    def add_uncertainty(self, item: str) -> "TechnicianLogBuilder":
        """Add an uncertainty item."""
        self.log.uncertain_about.append(item)
        return self

    def add_info_needed(self, item: str) -> "TechnicianLogBuilder":
        """Add additional info needed."""
        self.log.additional_info_needed.append(item)
        return self

    def set_recommendation(self, action: str) -> "TechnicianLogBuilder":
        """Set recommended action."""
        self.log.recommended_action = action
        return self

    def add_safety_warning(self, warning: str) -> "TechnicianLogBuilder":
        """Add a safety warning."""
        self.log.safety_warnings.append(warning)
        return self

    def add_prerequisite(self, prereq: str) -> "TechnicianLogBuilder":
        """Add a prerequisite."""
        self.log.prerequisites.append(prereq)
        return self

    def add_refinement(
        self,
        reason: str,
        action: str,
        iteration: int,
        improvement: Optional[str] = None
    ) -> "TechnicianLogBuilder":
        """Add a refinement record."""
        self.log.refinements_made.append(RefinementRecord(
            reason=reason,
            action=action,
            iteration=iteration,
            improvement=improvement
        ))
        return self

    def add_error_code(self, code: str) -> "TechnicianLogBuilder":
        """Add an identified error code."""
        if code not in self.log.error_codes_identified:
            self.log.error_codes_identified.append(code)
        return self

    def add_related_error(self, code: str) -> "TechnicianLogBuilder":
        """Add a related error code."""
        if code not in self.log.related_errors:
            self.log.related_errors.append(code)
        return self

    def add_equipment(self, equipment: str) -> "TechnicianLogBuilder":
        """Add mentioned equipment."""
        if equipment not in self.log.equipment_mentioned:
            self.log.equipment_mentioned.append(equipment)
        return self

    def build(self) -> TechnicianLog:
        """Build and return the TechnicianLog."""
        return self.log


# In-memory store for request logs
_technician_logs: Dict[str, TechnicianLog] = {}


def store_technician_log(log: TechnicianLog):
    """Store a technician log for later retrieval."""
    _technician_logs[log.request_id] = log
    # Keep only last 1000 logs
    if len(_technician_logs) > 1000:
        oldest_key = next(iter(_technician_logs))
        del _technician_logs[oldest_key]


def get_technician_log(request_id: str) -> Optional[TechnicianLog]:
    """Retrieve a stored technician log."""
    return _technician_logs.get(request_id)


def get_log_builder(request_id: str) -> TechnicianLogBuilder:
    """
    Factory function to get a TechnicianLogBuilder instance.

    Args:
        request_id: Unique request identifier

    Returns:
        TechnicianLogBuilder instance
    """
    return TechnicianLogBuilder(request_id)
