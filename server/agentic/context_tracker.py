"""
Context Tracker - Monitor context flow between pipeline stages.

Part of P0 Observability Enhancement (OBSERVABILITY_IMPROVEMENT_PLAN.md).

Tracks how context (queries, search results, scraped content, findings, synthesis)
flows through the agentic pipeline, enabling:
- Context size monitoring (token/char counts)
- Information bottleneck detection
- Data flow visualization
- Debugging context loss between stages

Created: 2026-01-02
"""

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ContextType(str, Enum):
    """Types of context that flow through the pipeline."""
    QUERY = "query"
    QUERY_ANALYSIS = "query_analysis"
    DECOMPOSED_QUESTIONS = "decomposed_questions"
    SEARCH_QUERIES = "search_queries"
    SEARCH_RESULTS = "search_results"
    SCRAPED_CONTENT = "scraped_content"
    FINDINGS = "findings"
    ENTITIES = "entities"
    VERIFICATION_RESULTS = "verification_results"
    SYNTHESIS = "synthesis"
    REFLECTION = "reflection"
    CRAG_EVALUATION = "crag_evaluation"
    SCRATCHPAD_STATE = "scratchpad_state"
    TEMPLATE = "template"
    SOURCES = "sources"


class PipelineStage(str, Enum):
    """Pipeline stages that process context."""
    INPUT = "input"
    QUERY_CLASSIFIER = "query_classifier"
    ANALYZER = "analyzer"
    PLANNER = "planner"
    HYDE = "hyde"
    SEARCHER = "searcher"
    SCRAPER = "scraper"
    CRAG = "crag"
    CONTEXT_CURATOR = "context_curator"
    INFORMATION_BOTTLENECK = "information_bottleneck"
    ENTITY_TRACKER = "entity_tracker"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"
    SELF_RAG = "self_rag"
    ADAPTIVE_REFINEMENT = "adaptive_refinement"
    OUTPUT = "output"


@dataclass
class ContextTransfer:
    """Record of a context transfer between pipeline stages."""
    request_id: str
    source_stage: str
    target_stage: str
    context_type: str
    token_count: int  # Estimated tokens (~chars/4)
    char_count: int
    item_count: int  # Number of sources, findings, questions, etc.
    content_hash: str  # For deduplication detection (first 8 chars of MD5)
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional detailed metrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "request_id": self.request_id,
            "source": self.source_stage,
            "target": self.target_stage,
            "type": self.context_type,
            "tokens": self.token_count,
            "chars": self.char_count,
            "items": self.item_count,
            "hash": self.content_hash,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ContextSnapshot:
    """Snapshot of context at a specific pipeline stage."""
    stage: str
    timestamp: datetime
    total_tokens: int
    total_chars: int
    context_types: Dict[str, int]  # Type -> item count
    content_preview: str  # First 200 chars for debugging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "total_tokens": self.total_tokens,
            "total_chars": self.total_chars,
            "context_types": self.context_types,
            "preview": self.content_preview
        }


class ContextFlowTracker:
    """
    Track all context transfers in a request pipeline.

    Usage:
        tracker = ContextFlowTracker("req-123")

        # Record context transfers
        tracker.record_transfer(
            source="analyzer",
            target="planner",
            content=analysis_result,
            context_type=ContextType.QUERY_ANALYSIS
        )

        # Get flow summary
        summary = tracker.get_flow_summary()
    """

    def __init__(
        self,
        request_id: str,
        emitter: Optional[Any] = None,
        verbose: bool = False
    ):
        self.request_id = request_id
        self.emitter = emitter
        self.verbose = verbose
        self.transfers: List[ContextTransfer] = []
        self.snapshots: Dict[str, ContextSnapshot] = {}

        # Aggregated metrics
        self._total_tokens_by_type: Dict[str, int] = {}
        self._transfers_by_stage: Dict[str, int] = {}
        self._start_time = datetime.now()

    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count from content."""
        if content is None:
            return 0
        text = str(content)
        return len(text) // 4  # Rough approximation

    def _calculate_char_count(self, content: Any) -> int:
        """Calculate character count from content."""
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content)
        elif isinstance(content, (list, tuple)):
            return sum(self._calculate_char_count(c) for c in content)
        elif isinstance(content, dict):
            return sum(self._calculate_char_count(v) for v in content.values())
        else:
            return len(str(content))

    def _calculate_item_count(self, content: Any) -> int:
        """Calculate item count from content."""
        if content is None:
            return 0
        if isinstance(content, (list, tuple)):
            return len(content)
        elif isinstance(content, dict):
            return len(content)
        else:
            return 1

    def _content_hash(self, content: Any) -> str:
        """Generate short hash for deduplication detection."""
        try:
            content_str = str(content)[:10000]  # Limit for hashing
            return hashlib.md5(content_str.encode()).hexdigest()[:8]
        except Exception:
            return "????????"

    def record_transfer(
        self,
        source: str,
        target: str,
        content: Any,
        context_type: Union[str, ContextType],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextTransfer:
        """
        Record a context transfer between pipeline stages.

        Args:
            source: Source stage name
            target: Target stage name
            content: The actual content being transferred
            context_type: Type of context (use ContextType enum)
            metadata: Additional metadata about the transfer

        Returns:
            The recorded ContextTransfer
        """
        # Convert enum to string if needed
        if isinstance(context_type, ContextType):
            context_type = context_type.value

        # Calculate metrics
        char_count = self._calculate_char_count(content)
        token_count = char_count // 4
        item_count = self._calculate_item_count(content)
        content_hash = self._content_hash(content)

        transfer = ContextTransfer(
            request_id=self.request_id,
            source_stage=source,
            target_stage=target,
            context_type=context_type,
            token_count=token_count,
            char_count=char_count,
            item_count=item_count,
            content_hash=content_hash,
            metadata=metadata or {}
        )

        # Store transfer
        self.transfers.append(transfer)

        # Update aggregated metrics
        self._total_tokens_by_type[context_type] = \
            self._total_tokens_by_type.get(context_type, 0) + token_count
        self._transfers_by_stage[source] = \
            self._transfers_by_stage.get(source, 0) + 1

        # Log transfer
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logger.log(
            log_level,
            f"[{self.request_id}] Context: {source} → {target} | "
            f"type={context_type} | items={item_count} | ~{token_count} tokens"
        )

        # Emit SSE event if available
        if self.emitter:
            try:
                self._emit_transfer_event(transfer)
            except Exception as e:
                logger.debug(f"Failed to emit context transfer event: {e}")

        return transfer

    def _emit_transfer_event(self, transfer: ContextTransfer):
        """Emit SSE event for context transfer (async wrapper)."""
        import asyncio
        try:
            from agentic.models import SearchEvent, EventType
            event = SearchEvent(
                event_type=EventType.CONTEXT_TRANSFER,
                request_id=self.request_id,
                data={
                    "source": transfer.source_stage,
                    "target": transfer.target_stage,
                    "type": transfer.context_type,
                    "tokens": transfer.token_count,
                    "items": transfer.item_count
                }
            )
            # Try to emit asynchronously
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.emitter.emit(event))
            else:
                loop.run_until_complete(self.emitter.emit(event))
        except Exception as e:
            logger.debug(f"SSE emit failed: {e}")

    def take_snapshot(
        self,
        stage: str,
        content: Any,
        context_types: Optional[Dict[str, int]] = None
    ) -> ContextSnapshot:
        """
        Take a snapshot of context at a specific stage.

        Useful for debugging what context is available at each stage.

        Args:
            stage: Pipeline stage name
            content: Current context content
            context_types: Optional breakdown by context type

        Returns:
            ContextSnapshot
        """
        total_chars = self._calculate_char_count(content)
        total_tokens = total_chars // 4

        # Generate preview
        content_str = str(content)
        preview = content_str[:200] + "..." if len(content_str) > 200 else content_str

        snapshot = ContextSnapshot(
            stage=stage,
            timestamp=datetime.now(),
            total_tokens=total_tokens,
            total_chars=total_chars,
            context_types=context_types or {"content": self._calculate_item_count(content)},
            content_preview=preview
        )

        self.snapshots[stage] = snapshot

        logger.debug(
            f"[{self.request_id}] Snapshot at {stage}: "
            f"~{total_tokens} tokens, {len(context_types or {})} types"
        )

        return snapshot

    def get_flow_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive flow summary for debugging.

        Returns:
            Summary dict suitable for response metadata
        """
        if not self.transfers:
            return {
                "total_transfers": 0,
                "total_tokens": 0,
                "flow_path": [],
                "by_type": {},
                "by_stage": {}
            }

        total_tokens = sum(t.token_count for t in self.transfers)
        duration_ms = int((datetime.now() - self._start_time).total_seconds() * 1000)

        # Build flow path (source→target pairs)
        flow_path = [f"{t.source_stage}→{t.target_stage}" for t in self.transfers]

        # Detect potential bottlenecks (large token drops)
        bottlenecks = []
        for i, transfer in enumerate(self.transfers):
            if i > 0 and transfer.token_count < self.transfers[i-1].token_count * 0.3:
                bottlenecks.append({
                    "stage": transfer.target_stage,
                    "reduction": round(1 - transfer.token_count / max(self.transfers[i-1].token_count, 1), 2)
                })

        return {
            "total_transfers": len(self.transfers),
            "total_tokens_transferred": total_tokens,
            "duration_ms": duration_ms,
            "flow_path": flow_path[:30],  # Limit for response size
            "by_type": dict(self._total_tokens_by_type),
            "by_stage": dict(self._transfers_by_stage),
            "bottlenecks": bottlenecks[:5] if bottlenecks else [],
            "snapshots": {k: v.to_dict() for k, v in self.snapshots.items()}
        }

    def get_transfers_to_stage(self, stage: str) -> List[ContextTransfer]:
        """Get all transfers targeting a specific stage."""
        return [t for t in self.transfers if t.target_stage == stage]

    def get_transfers_from_stage(self, stage: str) -> List[ContextTransfer]:
        """Get all transfers originating from a specific stage."""
        return [t for t in self.transfers if t.source_stage == stage]

    def get_transfers_by_type(self, context_type: str) -> List[ContextTransfer]:
        """Get all transfers of a specific context type."""
        return [t for t in self.transfers if t.context_type == context_type]

    def detect_duplicate_content(self) -> List[Dict[str, Any]]:
        """
        Detect potential duplicate content transfers.

        Returns list of duplicate pairs based on content hash.
        """
        hash_to_transfers: Dict[str, List[ContextTransfer]] = {}
        for transfer in self.transfers:
            if transfer.content_hash not in hash_to_transfers:
                hash_to_transfers[transfer.content_hash] = []
            hash_to_transfers[transfer.content_hash].append(transfer)

        duplicates = []
        for hash_val, transfers in hash_to_transfers.items():
            if len(transfers) > 1:
                duplicates.append({
                    "hash": hash_val,
                    "count": len(transfers),
                    "stages": [t.target_stage for t in transfers]
                })

        return duplicates

    def get_total_tokens_at_stage(self, stage: str) -> int:
        """Calculate total tokens received at a stage."""
        return sum(t.token_count for t in self.get_transfers_to_stage(stage))

    def export_for_debugging(self) -> Dict[str, Any]:
        """Export full transfer history for detailed debugging."""
        return {
            "request_id": self.request_id,
            "start_time": self._start_time.isoformat(),
            "transfers": [t.to_log_dict() for t in self.transfers],
            "snapshots": {k: v.to_dict() for k, v in self.snapshots.items()},
            "summary": self.get_flow_summary()
        }


def get_context_tracker(
    request_id: str,
    emitter: Optional[Any] = None,
    verbose: bool = False
) -> ContextFlowTracker:
    """
    Factory function to get a ContextFlowTracker instance.

    Args:
        request_id: Unique request identifier
        emitter: Optional SSE event emitter
        verbose: Enable verbose logging

    Returns:
        ContextFlowTracker instance
    """
    return ContextFlowTracker(request_id, emitter, verbose)
