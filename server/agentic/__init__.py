"""
Agentic Search Module for memOS

Implements multi-agent search with ReAct pattern:
- Orchestrator: Routes queries to appropriate pipelines
- Planner: Decomposes queries into search terms
- Searcher: Executes web searches (Brave API, DuckDuckGo fallback)
- Verifier: Cross-checks facts and validates claims
- Synthesizer: Combines results into coherent answers

This module is isolated from core memOS services and can be
enabled/disabled independently.
"""

from .orchestrator import AgenticOrchestrator
from .models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    VerificationResult,
    AgentAction,
    SearchState
)
from .events import (
    EventType,
    SearchEvent,
    EventEmitter,
    EventManager,
    get_event_manager
)
from . import events

__all__ = [
    "AgenticOrchestrator",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "VerificationResult",
    "AgentAction",
    "SearchState",
    "EventType",
    "SearchEvent",
    "EventEmitter",
    "EventManager",
    "get_event_manager",
    "events"
]

__version__ = "0.1.0"
