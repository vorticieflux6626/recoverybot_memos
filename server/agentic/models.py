"""
Pydantic models for Agentic Search System

Defines all data structures used in the multi-agent search pipeline.
Follows the unified response format from UNIFIED_ARCHITECTURE_RECOMMENDATIONS.md
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
import uuid


class VerificationLevel(str, Enum):
    """Level of fact verification to apply"""
    NONE = "none"           # Skip verification (fastest)
    STANDARD = "standard"   # Basic cross-checking
    STRICT = "strict"       # Thorough multi-source verification


class SearchMode(str, Enum):
    """Search mode determining iteration behavior"""
    FIXED = "fixed"         # Fixed number of iterations (default)
    ADAPTIVE = "adaptive"   # Continue until information found or leads exhausted
    EXHAUSTIVE = "exhaustive"  # Search all possible leads regardless of findings


class ActionType(str, Enum):
    """Types of actions an agent can take"""
    ANALYZE = "analyze"     # Analyze if search is needed
    PLAN = "plan"           # Create search plan
    SEARCH = "search"       # Execute web search
    REFINE = "refine"       # Refine search based on results
    VERIFY = "verify"       # Verify claims
    SYNTHESIZE = "synthesize"  # Combine results
    DONE = "done"           # Finished processing


class ConfidenceLevel(str, Enum):
    """Confidence in the synthesized result"""
    HIGH = "high"           # Multiple sources agree
    MEDIUM = "medium"       # Some sources, minor conflicts
    LOW = "low"             # Limited sources or conflicts
    UNKNOWN = "unknown"     # Could not determine


class ReasoningComplexity(str, Enum):
    """
    Level of reasoning complexity requiring different model capabilities.
    Used to determine whether to use a thinking model (DeepSeek R1) for synthesis.
    """
    SIMPLE = "simple"           # Straightforward lookup/retrieval
    MODERATE = "moderate"       # Standard synthesis and summarization
    COMPLEX = "complex"         # Multi-step reasoning, comparisons
    EXPERT = "expert"           # Technical troubleshooting, debugging, deep analysis


# Request Models

class SearchRequest(BaseModel):
    """Request for agentic search"""
    query: str = Field(..., description="The user's search query", min_length=3)
    user_id: Optional[str] = Field(None, description="User ID for personalization and caching")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context (conversation history, preferences)"
    )
    max_iterations: int = Field(
        default=10,  # Increased from 5 for thorough multi-direction exploration
        ge=1,
        le=50,  # Allows extensive search when needed
        description="Maximum ReAct loop iterations (higher for ADAPTIVE/EXHAUSTIVE modes)"
    )
    search_mode: SearchMode = Field(
        default=SearchMode.ADAPTIVE,
        description="Search mode: FIXED (stop at max_iterations), ADAPTIVE (stop when satisfied), EXHAUSTIVE (search all leads)"
    )
    analyze_query: bool = Field(
        default=True,
        description="Use LLM to analyze query and determine if web search is beneficial"
    )
    verification_level: VerificationLevel = Field(
        default=VerificationLevel.STANDARD,
        description="Level of fact verification"
    )
    cache_results: bool = Field(
        default=True,
        description="Whether to cache results in memory service"
    )
    min_sources: int = Field(
        default=5,  # Increased from 3 for better corpus quality
        ge=1,
        le=30,
        description="Minimum number of sources to consult before stopping (for ADAPTIVE mode)"
    )
    max_sources: int = Field(
        default=25,  # Increased from 15 for comprehensive research
        ge=3,
        le=50,
        description="Maximum number of sources to consult"
    )
    min_confidence: float = Field(
        default=0.70,  # Minimum confidence threshold for sufficient corpus
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to consider corpus sufficient"
    )
    force_thinking_model: bool = Field(
        default=False,
        description="Force use of thinking model (DeepSeek R1) for synthesis regardless of query analysis"
    )
    max_scrape_refinements: int = Field(
        default=3,  # Allow multiple refinement cycles
        ge=0,
        le=10,
        description="Maximum scrape refinement cycles to improve corpus quality"
    )
    # Context utilization configuration
    max_urls_to_scrape: int = Field(
        default=15,  # Increased from 8 to maximize source utilization
        ge=1,
        le=30,
        description="Maximum URLs to scrape for content (higher uses more context)"
    )
    max_content_per_source: int = Field(
        default=10000,  # 10K chars per source
        ge=1000,
        le=50000,
        description="Maximum characters to use from each scraped source"
    )
    max_synthesis_context: int = Field(
        default=48000,  # Expanded from 24K to utilize 32K context window
        ge=8000,
        le=100000,
        description="Maximum total characters for synthesis prompt (should be ~1.5x context window)"
    )


class SimpleSearchRequest(BaseModel):
    """Request for simple (non-agentic) search"""
    query: str = Field(..., description="Search query", min_length=3)
    max_results: int = Field(default=5, ge=1, le=20)


# Internal State Models

class WebSearchResult(BaseModel):
    """Individual web search result"""
    title: str
    url: str
    snippet: str
    source_domain: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class QueryAnalysis(BaseModel):
    """Result of LLM analyzing the user query"""
    requires_search: bool = Field(default=True, description="Whether web search would be beneficial")
    search_reasoning: str = Field(default="", description="Reasoning for search decision")
    query_type: str = Field(default="informational", description="Type: factual, opinion, local_service, crisis, etc.")
    key_topics: List[str] = Field(default_factory=list, description="Key topics to research")
    suggested_queries: List[str] = Field(default_factory=list, description="Initial search queries")
    priority_domains: List[str] = Field(default_factory=list, description="Domains to prioritize")
    estimated_complexity: str = Field(default="medium", description="low, medium, high complexity")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Thinking model classification (NEW - December 2025)
    requires_thinking_model: bool = Field(
        default=False,
        description="Whether this query requires a thinking/reasoning model (DeepSeek R1) for synthesis"
    )
    reasoning_complexity: str = Field(
        default="moderate",
        description="Reasoning complexity: simple, moderate, complex, expert"
    )
    thinking_model_reasoning: str = Field(
        default="",
        description="Why a thinking model is/isn't needed"
    )


class SearchPlan(BaseModel):
    """Multi-step search plan generated by planner"""
    original_query: str
    decomposed_questions: List[str] = Field(default_factory=list)
    search_phases: List[Dict[str, Any]] = Field(default_factory=list)
    priority_order: List[int] = Field(default_factory=list)
    fallback_strategies: List[str] = Field(default_factory=list)
    estimated_iterations: int = 3
    reasoning: str = ""


class AgentAction(BaseModel):
    """Action decided by an agent"""
    type: ActionType
    queries: List[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    next_action: Optional[str] = None
    refinement_needed: bool = False


class VerificationResult(BaseModel):
    """Result of fact verification"""
    claim: str
    verified: bool
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)


class SearchState(BaseModel):
    """State maintained during agentic search"""
    query: str
    search_queries: List[str] = Field(default_factory=list)
    executed_queries: List[str] = Field(default_factory=list)
    pending_queries: List[str] = Field(default_factory=list)
    raw_results: List[WebSearchResult] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    verified_claims: List[VerificationResult] = Field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 5
    search_mode: str = "adaptive"

    # Query analysis state
    query_analysis: Optional[QueryAnalysis] = None
    search_plan: Optional[SearchPlan] = None

    # GAP-3 fix: Directive propagation fields (propagated from query_analysis)
    key_topics: List[str] = Field(default_factory=list, description="Key topics from analyzer for downstream use")
    priority_domains: List[str] = Field(default_factory=list, description="Priority domains from analyzer for source validation")
    active_constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Active constraints for verification gate")
    directive_source: str = Field(default="", description="Source of directives (e.g., 'analyzer')")

    # Tracking for adaptive search
    sources_consulted: int = 0
    unique_domains: List[str] = Field(default_factory=list)
    leads_exhausted: bool = False
    information_sufficient: bool = False
    refinement_attempts: int = 0
    max_refinements: int = 3

    # Search progress
    search_phases_completed: List[str] = Field(default_factory=list)
    current_phase: str = "initial"

    def add_results(self, results: List[WebSearchResult]):
        """Add search results to state"""
        for result in results:
            self.raw_results.append(result)
            self.sources_consulted += 1
            if result.source_domain not in self.unique_domains:
                self.unique_domains.append(result.source_domain)

    def add_claims(self, claims: List[str]):
        """Add extracted claims for verification"""
        self.claims.extend(claims)

    def has_sufficient_sources(self, min_sources: int) -> bool:
        """Check if we have enough sources"""
        return len(self.unique_domains) >= min_sources

    def can_continue_search(self, max_sources: int) -> bool:
        """Check if we should continue searching"""
        if self.leads_exhausted:
            return False
        if self.sources_consulted >= max_sources:
            return False
        if self.search_mode == "fixed" and self.iteration >= self.max_iterations:
            return False
        return True

    def mark_query_executed(self, query: str):
        """Mark a query as executed"""
        if query not in self.executed_queries:
            self.executed_queries.append(query)
        if query in self.pending_queries:
            self.pending_queries.remove(query)

    def add_pending_queries(self, queries: List[str]):
        """Add new queries to pending list (avoiding duplicates)"""
        for q in queries:
            if q not in self.executed_queries and q not in self.pending_queries:
                self.pending_queries.append(q)


# Response Models

class SearchMeta(BaseModel):
    """Metadata about the search execution"""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0.0"
    iterations: int = 0
    queries_executed: int = 0
    sources_consulted: int = 0
    execution_time_ms: int = 0
    cache_hit: bool = False
    # Semantic cache fields (Phase 2 optimization)
    semantic_match: bool = False
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    # Unified orchestrator enhancement metadata
    enhancement_metadata: Optional[Dict[str, Any]] = None


class SearchResultData(BaseModel):
    """Data portion of search response"""
    synthesized_context: str = Field(..., description="The synthesized answer")
    sources: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Sources used in synthesis"
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Queries executed during search"
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the result"
    )
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    verification_status: str = "unverified"
    search_trace: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Trace of agent actions for debugging"
    )


class SearchResponse(BaseModel):
    """
    Unified response format for agentic search.
    Follows the envelope pattern from UNIFIED_ARCHITECTURE_RECOMMENDATIONS.md
    """
    success: bool = True
    data: SearchResultData
    meta: SearchMeta
    errors: List[Dict[str, str]] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "synthesized_context": "The recommended approach for solving this problem includes...",
                    "sources": [
                        {"title": "Official Documentation", "url": "https://docs.example.com/..."}
                    ],
                    "search_queries": [
                        "technical problem solution guide",
                        "best practices implementation"
                    ],
                    "confidence_score": 0.85,
                    "confidence_level": "high",
                    "verification_status": "verified"
                },
                "meta": {
                    "timestamp": "2025-12-24T00:00:00Z",
                    "request_id": "abc-123",
                    "version": "1.0.0",
                    "iterations": 2,
                    "queries_executed": 3
                },
                "errors": []
            }
        }
    )


class SearchResult(BaseModel):
    """Legacy compatibility model"""
    query: str
    search_queries: List[str]
    raw_results: List[Dict[str, Any]]
    synthesis: str
    execution_time: float
    iterations: int
    confidence: float = 0.5
    verified: bool = False
