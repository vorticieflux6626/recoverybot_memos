"""
Agentic Search Scratchpad - Shared Working Memory for Multi-Agent Pipeline

This module implements a "blackboard" pattern where all agents in the search
pipeline can read and write to a shared scratchpad. This enables:

1. **Intelligent Direction**: Higher-order process can set goals and track completion
2. **Information Persistence**: Findings survive between agent calls
3. **Gap Detection**: Agents can see what's missing and prioritize accordingly
4. **Contradiction Tracking**: Conflicting information is flagged for resolution
5. **Task Completion Criteria**: Explicit conditions for when search is "done"

Architecture:
    User Query
         ↓
    ┌─────────────────────────────────────────────┐
    │              SCRATCHPAD (memOS)              │
    │  ┌─────────────────────────────────────┐    │
    │  │ Mission:                            │    │
    │  │   - Original query                  │    │
    │  │   - Decomposed questions            │    │
    │  │   - Completion criteria per Q       │    │
    │  └─────────────────────────────────────┘    │
    │  ┌─────────────────────────────────────┐    │
    │  │ Working Notes:                      │    │
    │  │   Q1: [answered] confidence: 0.9    │    │
    │  │   Q2: [partial] need: cost info     │    │
    │  │   Q3: [unanswered] searching...     │    │
    │  └─────────────────────────────────────┘    │
    │  ┌─────────────────────────────────────┐    │
    │  │ Findings:                           │    │
    │  │   - Fact A (source: X, conf: 0.8)   │    │
    │  │   - Fact B (source: Y, conf: 0.7)   │    │
    │  │   - CONFLICT: A vs C (investigate)  │    │
    │  └─────────────────────────────────────┘    │
    │  ┌─────────────────────────────────────┐    │
    │  │ Next Actions:                       │    │
    │  │   1. Search for Q3 specifics        │    │
    │  │   2. Resolve A vs C conflict        │    │
    │  │   3. Verify cost information        │    │
    │  └─────────────────────────────────────┘    │
    └─────────────────────────────────────────────┘
         ↑↓              ↑↓              ↑↓
    [Analyzer]      [Searcher]      [Synthesizer]

Usage:
    # Initialize scratchpad for a new search
    scratchpad = AgenticScratchpad.create(
        query="What are FDA-approved OUD medications?",
        request_id="abc123"
    )

    # Analyzer decomposes and sets completion criteria
    await scratchpad.set_mission(
        decomposed_questions=[
            "What medications are FDA-approved for OUD?",
            "What are their mechanisms of action?",
            "Which have extended-release formulations?"
        ],
        completion_criteria={
            "q1": "List at least 3 medications with FDA approval status",
            "q2": "Explain mechanism for each medication",
            "q3": "Identify ER formulations with brand names"
        }
    )

    # Searcher adds findings
    await scratchpad.add_finding(
        question_id="q1",
        fact="Methadone is FDA-approved for OUD",
        source="fda.gov",
        confidence=0.95
    )

    # Check if we're done
    status = await scratchpad.get_completion_status()
    # Returns: {"q1": "answered", "q2": "partial", "q3": "unanswered", "overall": 0.45}
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
import json
import logging

logger = logging.getLogger("agentic.scratchpad")


class QuestionStatus(str, Enum):
    """Status of a decomposed question"""
    UNANSWERED = "unanswered"
    SEARCHING = "searching"
    PARTIAL = "partial"      # Some info found but incomplete
    ANSWERED = "answered"
    CONFLICT = "conflict"    # Contradictory information found


class FindingType(str, Enum):
    """Type of finding"""
    FACT = "fact"
    STATISTIC = "statistic"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    CONTACT = "contact"
    CONTRADICTION = "contradiction"


class ScratchpadFinding(BaseModel):
    """A single finding/fact discovered during search"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question_id: str  # Which decomposed question this addresses
    finding_type: FindingType = FindingType.FACT
    content: str
    source_url: str
    source_title: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verified: bool = False
    conflicts_with: List[str] = Field(default_factory=list)  # IDs of conflicting findings


class QuestionProgress(BaseModel):
    """Progress tracking for a single decomposed question"""
    question_id: str
    question_text: str
    completion_criteria: str
    status: QuestionStatus = QuestionStatus.UNANSWERED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    findings: List[str] = Field(default_factory=list)  # Finding IDs
    gaps: List[str] = Field(default_factory=list)  # What's still missing
    notes: str = ""


class AgentNote(BaseModel):
    """A note left by an agent for other agents to read"""
    agent: str  # analyzer, planner, searcher, verifier, synthesizer
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action_taken: str
    observation: str
    recommendation: str = ""
    for_agent: Optional[str] = None  # Specific agent this note is for


class AgenticScratchpad(BaseModel):
    """
    Shared working memory for the agentic search pipeline.

    This is the central "blackboard" that all agents read from and write to,
    enabling coordinated multi-agent search with explicit task tracking.
    """

    # Identity
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Mission
    original_query: str
    user_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

    # Decomposed Questions & Progress
    questions: Dict[str, QuestionProgress] = Field(default_factory=dict)

    # Findings Repository
    findings: Dict[str, ScratchpadFinding] = Field(default_factory=dict)

    # Contradictions to resolve
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)

    # Agent Communication
    agent_notes: List[AgentNote] = Field(default_factory=list)

    # Search History
    queries_executed: List[str] = Field(default_factory=list)
    urls_scraped: List[str] = Field(default_factory=list)
    sources_consulted: int = 0

    # Next Actions Queue (priority ordered)
    next_actions: List[Dict[str, Any]] = Field(default_factory=list)

    # Completion Tracking
    is_complete: bool = False
    completion_reason: str = ""
    overall_confidence: float = 0.0

    @classmethod
    def create(cls, query: str, request_id: Optional[str] = None, user_id: Optional[str] = None) -> "AgenticScratchpad":
        """Factory method to create a new scratchpad for a search"""
        return cls(
            request_id=request_id or str(uuid.uuid4()),
            original_query=query,
            user_id=user_id
        )

    # ========================================
    # MISSION SETUP (Called by Analyzer)
    # ========================================

    def set_mission(
        self,
        decomposed_questions: List[str],
        completion_criteria: Dict[str, str]
    ) -> None:
        """
        Set the mission by decomposing the query into sub-questions
        with explicit completion criteria for each.

        Args:
            decomposed_questions: List of sub-questions to answer
            completion_criteria: Dict mapping question ID to completion criteria
        """
        for i, question in enumerate(decomposed_questions):
            q_id = f"q{i+1}"
            self.questions[q_id] = QuestionProgress(
                question_id=q_id,
                question_text=question,
                completion_criteria=completion_criteria.get(q_id, "Find relevant information"),
                status=QuestionStatus.UNANSWERED
            )

        self._add_agent_note(
            agent="analyzer",
            action_taken=f"Decomposed query into {len(decomposed_questions)} questions",
            observation=f"Questions: {decomposed_questions}",
            recommendation="Begin searching for Q1 first"
        )
        self._touch()

    # ========================================
    # FINDINGS MANAGEMENT (Called by Searcher/Scraper)
    # ========================================

    def add_finding(
        self,
        question_id: str,
        content: str,
        source_url: str,
        source_title: str = "",
        finding_type: FindingType = FindingType.FACT,
        confidence: float = 0.5
    ) -> str:
        """
        Add a new finding/fact discovered during search.

        Returns:
            finding_id: The ID of the new finding
        """
        finding = ScratchpadFinding(
            question_id=question_id,
            finding_type=finding_type,
            content=content,
            source_url=source_url,
            source_title=source_title,
            confidence=confidence
        )

        self.findings[finding.id] = finding

        # Update question progress
        if question_id in self.questions:
            q = self.questions[question_id]
            q.findings.append(finding.id)
            q.status = QuestionStatus.PARTIAL
            q.confidence = max(q.confidence, confidence)

        self._touch()
        return finding.id

    def add_contradiction(
        self,
        finding_id_1: str,
        finding_id_2: str,
        description: str
    ) -> None:
        """Record a contradiction between two findings"""
        self.contradictions.append({
            "finding_1": finding_id_1,
            "finding_2": finding_id_2,
            "description": description,
            "resolved": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Mark findings as conflicting
        if finding_id_1 in self.findings:
            self.findings[finding_id_1].conflicts_with.append(finding_id_2)
        if finding_id_2 in self.findings:
            self.findings[finding_id_2].conflicts_with.append(finding_id_1)

        # Add to next actions
        self.add_next_action(
            action_type="resolve_contradiction",
            priority=1,
            details={"finding_1": finding_id_1, "finding_2": finding_id_2, "description": description}
        )

        self._touch()

    # ========================================
    # PROGRESS TRACKING
    # ========================================

    def mark_question_answered(
        self,
        question_id: str,
        confidence: float = 0.8,
        notes: str = ""
    ) -> None:
        """Mark a question as fully answered"""
        if question_id in self.questions:
            q = self.questions[question_id]
            q.status = QuestionStatus.ANSWERED
            q.confidence = confidence
            q.notes = notes
            self._touch()

    def mark_question_partial(
        self,
        question_id: str,
        gaps: List[str],
        confidence: float = 0.5
    ) -> None:
        """Mark a question as partially answered with specific gaps"""
        if question_id in self.questions:
            q = self.questions[question_id]
            q.status = QuestionStatus.PARTIAL
            q.gaps = gaps
            q.confidence = confidence

            # Add gap-filling to next actions
            for gap in gaps:
                self.add_next_action(
                    action_type="fill_gap",
                    priority=2,
                    details={"question_id": question_id, "gap": gap}
                )

            self._touch()

    def get_completion_status(self) -> Dict[str, Any]:
        """
        Get the current completion status of the search.

        Returns:
            Dict with status per question and overall completion percentage
        """
        if not self.questions:
            return {"overall": 0.0, "questions": {}}

        status = {}
        total_confidence = 0.0

        for q_id, q in self.questions.items():
            status[q_id] = {
                "status": q.status.value,
                "confidence": q.confidence,
                "findings_count": len(q.findings),
                "gaps": q.gaps
            }
            total_confidence += q.confidence

        overall = total_confidence / len(self.questions)

        return {
            "overall": overall,
            "is_complete": overall >= 0.7 and all(
                q.status in [QuestionStatus.ANSWERED, QuestionStatus.PARTIAL]
                for q in self.questions.values()
            ),
            "questions": status,
            "unresolved_contradictions": len([c for c in self.contradictions if not c.get("resolved")])
        }

    def get_unanswered_questions(self) -> List[QuestionProgress]:
        """Get questions that still need answers"""
        return [
            q for q in self.questions.values()
            if q.status in [QuestionStatus.UNANSWERED, QuestionStatus.SEARCHING]
        ]

    def get_gaps(self) -> List[Dict[str, Any]]:
        """Get all information gaps across all questions"""
        gaps = []
        for q_id, q in self.questions.items():
            for gap in q.gaps:
                gaps.append({
                    "question_id": q_id,
                    "question": q.question_text,
                    "gap": gap
                })
        return gaps

    # ========================================
    # NEXT ACTIONS QUEUE
    # ========================================

    def add_next_action(
        self,
        action_type: str,
        priority: int = 5,
        details: Dict[str, Any] = None
    ) -> None:
        """Add an action to the queue (lower priority = more urgent)"""
        self.next_actions.append({
            "type": action_type,
            "priority": priority,
            "details": details or {},
            "added_at": datetime.now(timezone.utc).isoformat()
        })
        # Sort by priority
        self.next_actions.sort(key=lambda x: x["priority"])
        self._touch()

    def get_next_action(self) -> Optional[Dict[str, Any]]:
        """Get and remove the highest priority action"""
        if self.next_actions:
            return self.next_actions.pop(0)
        return None

    def peek_next_actions(self, n: int = 3) -> List[Dict[str, Any]]:
        """Preview the next N actions without removing them"""
        return self.next_actions[:n]

    # ========================================
    # AGENT COMMUNICATION
    # ========================================

    def _add_agent_note(
        self,
        agent: str,
        action_taken: str,
        observation: str,
        recommendation: str = "",
        for_agent: Optional[str] = None
    ) -> None:
        """Internal method to add agent notes"""
        self.agent_notes.append(AgentNote(
            agent=agent,
            action_taken=action_taken,
            observation=observation,
            recommendation=recommendation,
            for_agent=for_agent
        ))

    def add_agent_note(
        self,
        agent: str,
        action_taken: str,
        observation: str,
        recommendation: str = "",
        for_agent: Optional[str] = None
    ) -> None:
        """Add a note from an agent for other agents to read"""
        self._add_agent_note(agent, action_taken, observation, recommendation, for_agent)
        self._touch()

    def get_notes_for_agent(self, agent: str) -> List[AgentNote]:
        """Get notes specifically for an agent, plus general notes"""
        return [
            note for note in self.agent_notes
            if note.for_agent is None or note.for_agent == agent
        ]

    def get_recent_notes(self, n: int = 5) -> List[AgentNote]:
        """Get the N most recent notes"""
        return sorted(self.agent_notes, key=lambda x: x.timestamp, reverse=True)[:n]

    # ========================================
    # SEARCH HISTORY
    # ========================================

    def record_search(self, query: str) -> None:
        """Record that a search query was executed"""
        if query not in self.queries_executed:
            self.queries_executed.append(query)
            self._touch()

    def record_scrape(self, url: str) -> None:
        """Record that a URL was scraped"""
        if url not in self.urls_scraped:
            self.urls_scraped.append(url)
            self.sources_consulted += 1
            self._touch()

    def has_searched(self, query: str) -> bool:
        """Check if a query has already been executed"""
        return query in self.queries_executed

    def has_scraped(self, url: str) -> bool:
        """Check if a URL has already been scraped"""
        return url in self.urls_scraped

    # ========================================
    # COMPLETION
    # ========================================

    def mark_complete(self, reason: str = "All questions answered") -> None:
        """Mark the search as complete"""
        status = self.get_completion_status()
        self.is_complete = True
        self.completion_reason = reason
        self.overall_confidence = status["overall"]
        self._touch()

    def should_continue(self, max_sources: int = 20) -> bool:
        """Determine if search should continue"""
        if self.is_complete:
            return False
        if self.sources_consulted >= max_sources:
            return False

        status = self.get_completion_status()

        # Continue if we have unanswered questions or gaps
        if status["overall"] < 0.7:
            return True
        if status["unresolved_contradictions"] > 0:
            return True
        if self.get_gaps():
            return True

        return False

    # ========================================
    # SERIALIZATION FOR memOS STORAGE
    # ========================================

    def to_memory_content(self) -> Dict[str, Any]:
        """Serialize to format suitable for memOS memory storage"""
        return {
            "type": "agentic_scratchpad",
            "request_id": self.request_id,
            "query": self.original_query,
            "questions": {
                q_id: {
                    "text": q.question_text,
                    "status": q.status.value,
                    "confidence": q.confidence,
                    "findings_count": len(q.findings),
                    "gaps": q.gaps
                }
                for q_id, q in self.questions.items()
            },
            "findings_count": len(self.findings),
            "sources_consulted": self.sources_consulted,
            "is_complete": self.is_complete,
            "overall_confidence": self.overall_confidence,
            "completion_status": self.get_completion_status()
        }

    @classmethod
    def from_memory_content(cls, content: Dict[str, Any]) -> "AgenticScratchpad":
        """Reconstruct scratchpad from memOS memory content"""
        scratchpad = cls(
            original_query=content.get("query", ""),
            request_id=content.get("request_id"),
            user_id=content.get("user_id")
        )

        # Restore questions
        for q_id, q_data in content.get("questions", {}).items():
            question = DecomposedQuestion(
                question_id=q_id,
                question_text=q_data.get("text", ""),
                status=QuestionStatus(q_data.get("status", "unanswered")),
                confidence=q_data.get("confidence", 0.0),
                gaps=q_data.get("gaps", [])
            )
            scratchpad.questions[q_id] = question

        scratchpad.sources_consulted = content.get("sources_consulted", 0)
        scratchpad.is_complete = content.get("is_complete", False)
        scratchpad.overall_confidence = content.get("overall_confidence", 0.0)

        logger.debug(f"Restored scratchpad {scratchpad.request_id[:8]} from memory")
        return scratchpad

    def to_context_for_agent(self, agent: str) -> str:
        """
        Generate a context string for an agent to read.
        This is what gets injected into the agent's prompt.
        """
        status = self.get_completion_status()

        context_parts = [
            f"=== SCRATCHPAD STATUS (Request: {self.request_id[:8]}) ===",
            f"Original Query: {self.original_query}",
            f"Overall Progress: {status['overall']:.0%}",
            "",
            "QUESTIONS STATUS:"
        ]

        for q_id, q in self.questions.items():
            emoji = {"answered": "✓", "partial": "◐", "unanswered": "○", "conflict": "⚠"}.get(q.status.value, "?")
            context_parts.append(f"  {emoji} [{q_id}] {q.question_text}")
            context_parts.append(f"      Status: {q.status.value} | Confidence: {q.confidence:.0%}")
            if q.gaps:
                context_parts.append(f"      Gaps: {', '.join(q.gaps)}")

        if self.contradictions:
            unresolved = [c for c in self.contradictions if not c.get("resolved")]
            if unresolved:
                context_parts.append("")
                context_parts.append(f"⚠ CONTRADICTIONS TO RESOLVE: {len(unresolved)}")
                for c in unresolved[:3]:
                    context_parts.append(f"  - {c['description']}")

        # Recent notes for this agent
        notes = self.get_notes_for_agent(agent)[-3:]
        if notes:
            context_parts.append("")
            context_parts.append("RECENT NOTES:")
            for note in notes:
                context_parts.append(f"  [{note.agent}]: {note.observation}")
                if note.recommendation:
                    context_parts.append(f"      → {note.recommendation}")

        # Next actions
        next_actions = self.peek_next_actions(3)
        if next_actions:
            context_parts.append("")
            context_parts.append("SUGGESTED NEXT ACTIONS:")
            for i, action in enumerate(next_actions, 1):
                context_parts.append(f"  {i}. {action['type']}: {action.get('details', {})}")

        return "\n".join(context_parts)

    # ========================================
    # INTERNAL HELPERS
    # ========================================

    def _touch(self) -> None:
        """Update the modified timestamp"""
        self.updated_at = datetime.now(timezone.utc)


# ========================================
# SCRATCHPAD MANAGER (integrates with memOS)
# ========================================

class ScratchpadManager:
    """
    Manages scratchpad lifecycle and persistence to memOS.

    This class handles:
    - Creating new scratchpads
    - Loading scratchpads from memOS memory
    - Persisting updates back to memOS
    - Cleaning up completed scratchpads
    """

    def __init__(self, memory_service=None):
        """
        Args:
            memory_service: Optional memOS memory service for persistence
        """
        self.memory_service = memory_service
        self._active_scratchpads: Dict[str, AgenticScratchpad] = {}

    def create(
        self,
        query: str,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> AgenticScratchpad:
        """Create a new scratchpad and register it"""
        scratchpad = AgenticScratchpad.create(query, request_id, user_id)
        self._active_scratchpads[scratchpad.request_id] = scratchpad
        logger.info(f"Created scratchpad {scratchpad.request_id[:8]} for query: {query[:50]}")
        return scratchpad

    def get(self, request_id: str) -> Optional[AgenticScratchpad]:
        """Get an active scratchpad by ID"""
        return self._active_scratchpads.get(request_id)

    async def persist(self, scratchpad: AgenticScratchpad) -> None:
        """Persist scratchpad state to memOS for session continuity"""
        if not self.memory_service:
            logger.debug(f"No memory service - scratchpad {scratchpad.request_id[:8]} not persisted")
            return

        try:
            from models.memory import MemoryType, MemoryPrivacyLevel
            import json

            memory_content = scratchpad.to_memory_content()

            # Store as procedural memory (how to do things / search results)
            await self.memory_service.store_memory(
                user_id=scratchpad.user_id or "system",
                content=json.dumps(memory_content),
                memory_type=MemoryType.PROCEDURAL,
                privacy_level=MemoryPrivacyLevel.MINIMAL,
                metadata={
                    "type": "scratchpad",
                    "request_id": scratchpad.request_id,
                    "query": scratchpad.original_query[:200],
                    "status": scratchpad.completion_status.value,
                    "confidence": scratchpad.overall_confidence
                },
                tags=["scratchpad", "agentic_search"],
                consent_given=True  # System-level storage
            )
            logger.debug(f"Persisted scratchpad {scratchpad.request_id[:8]} to memOS")
        except Exception as e:
            logger.warning(f"Failed to persist scratchpad: {e}")

    async def load_from_memory(self, request_id: str) -> Optional[AgenticScratchpad]:
        """Load a scratchpad from memOS by request_id"""
        if not self.memory_service:
            return None

        try:
            import json

            # Search for scratchpad in memory
            results = await self.memory_service.search_memories(
                user_id="system",
                query=f"scratchpad request_id:{request_id}",
                limit=1,
                memory_types=["procedural"]
            )

            if results and len(results) > 0:
                memory = results[0]
                content = json.loads(memory.content)

                # Reconstruct scratchpad from memory content
                scratchpad = AgenticScratchpad.from_memory_content(content)
                self._active_scratchpads[request_id] = scratchpad
                logger.info(f"Loaded scratchpad {request_id[:8]} from memOS")
                return scratchpad

        except Exception as e:
            logger.warning(f"Failed to load scratchpad from memOS: {e}")

        return None

    def complete(self, request_id: str, reason: str = "Search complete") -> Optional[AgenticScratchpad]:
        """Mark a scratchpad as complete and archive it"""
        scratchpad = self._active_scratchpads.get(request_id)
        if scratchpad:
            scratchpad.mark_complete(reason)
            # Could persist final state here
            del self._active_scratchpads[request_id]
            logger.info(f"Completed scratchpad {request_id[:8]}: {reason}")
        return scratchpad

    def cleanup_old(self, max_age_minutes: int = 30) -> int:
        """Remove scratchpads older than max_age_minutes"""
        now = datetime.now(timezone.utc)
        to_remove = []

        for req_id, sp in self._active_scratchpads.items():
            age = (now - sp.created_at).total_seconds() / 60
            if age > max_age_minutes:
                to_remove.append(req_id)

        for req_id in to_remove:
            del self._active_scratchpads[req_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old scratchpads")

        return len(to_remove)
