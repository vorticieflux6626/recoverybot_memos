"""
LLM Call Logger - Comprehensive LLM invocation tracking for observability.

Part of P1 Observability Enhancement (OBSERVABILITY_IMPROVEMENT_PLAN.md).

Provides detailed logging for all LLM calls in the agentic pipeline:
- Prompt/response token counts
- Timing metrics (latency, TTFT)
- Model selection and fallbacks
- Parse success/failure tracking
- Optional full prompt/response capture for debugging

Created: 2026-01-02
"""

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Generator

logger = logging.getLogger(__name__)


class LLMOperation(str, Enum):
    """Types of LLM operations in the pipeline."""
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    EVALUATION = "evaluation"
    REFLECTION = "reflection"
    CLASSIFICATION = "classification"
    DECOMPOSITION = "decomposition"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    EMBEDDING = "embedding"


@dataclass
class LLMCall:
    """Complete record of an LLM invocation."""
    request_id: str
    agent_name: str
    operation: str  # analysis, synthesis, verification, evaluation, reflection
    model: str

    # Input metrics
    prompt_template: str  # Template name or identifier
    prompt_length_chars: int
    prompt_length_tokens: int  # Estimated (~chars/4)
    input_context_items: int  # Number of sources, findings, etc.

    # Output metrics
    response_length_chars: int = 0
    response_length_tokens: int = 0
    response_truncated: bool = False

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    latency_ms: int = 0
    time_to_first_token_ms: Optional[int] = None

    # Quality indicators
    parse_success: bool = True  # Did JSON/structured output parse?
    fallback_used: bool = False  # Did we fall back to different model/provider?
    error_message: Optional[str] = None

    # Debug (opt-in)
    full_prompt: Optional[str] = None  # Only when verbose enabled
    full_response: Optional[str] = None  # Only when verbose enabled

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finalize(self, response: str = "", parse_success: bool = True, error: str = None):
        """Finalize the call record after completion."""
        self.end_time = datetime.now()
        self.latency_ms = int((self.end_time - self.start_time).total_seconds() * 1000)
        self.response_length_chars = len(response) if response else 0
        self.response_length_tokens = self.response_length_chars // 4
        self.parse_success = parse_success
        self.error_message = error

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "request_id": self.request_id,
            "agent": self.agent_name,
            "operation": self.operation,
            "model": self.model,
            "prompt_template": self.prompt_template,
            "prompt_tokens": self.prompt_length_tokens,
            "response_tokens": self.response_length_tokens,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.time_to_first_token_ms,
            "parse_success": self.parse_success,
            "fallback_used": self.fallback_used,
            "error": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for SSE events."""
        return {
            "agent": self.agent_name,
            "operation": self.operation,
            "model": self.model,
            "in_tokens": self.prompt_length_tokens,
            "out_tokens": self.response_length_tokens,
            "latency_ms": self.latency_ms,
            "success": self.parse_success and not self.error_message
        }


class LLMCallLogger:
    """
    Log all LLM calls with optional verbose mode.

    Usage:
        llm_logger = LLMCallLogger(request_id="req-123", verbose=False)

        # Using context manager
        async with llm_logger.track_call(
            agent_name="synthesizer",
            operation="synthesis",
            model="qwen3:8b",
            prompt=full_prompt
        ) as call:
            response = await call_ollama(full_prompt)
            call.finalize(response=response)

        # Get summary
        summary = llm_logger.get_call_summary()
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
        self.calls: List[LLMCall] = []

        # Aggregated metrics
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_latency_ms = 0
        self._calls_by_agent: Dict[str, int] = {}
        self._calls_by_operation: Dict[str, int] = {}

    @asynccontextmanager
    async def track_call(
        self,
        agent_name: str,
        operation: str,
        model: str,
        prompt: str,
        prompt_template: str = "custom",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Async context manager for tracking LLM calls.

        Args:
            agent_name: Name of the agent making the call
            operation: Type of operation (use LLMOperation enum)
            model: Model name/identifier
            prompt: The full prompt text
            prompt_template: Name of the prompt template used
            metadata: Additional metadata

        Yields:
            LLMCall object that should be finalized with response

        Example:
            async with llm_logger.track_call(...) as call:
                response = await ollama_generate(prompt)
                call.finalize(response=response, parse_success=True)
        """
        # Count context items (sources, findings, etc.)
        context_items = (
            prompt.count("[Source") +
            prompt.count("Finding:") +
            prompt.count("Question:") +
            prompt.count("Document:")
        )

        call = LLMCall(
            request_id=self.request_id,
            agent_name=agent_name,
            operation=operation,
            model=model,
            prompt_template=prompt_template,
            prompt_length_chars=len(prompt),
            prompt_length_tokens=len(prompt) // 4,
            input_context_items=context_items,
            full_prompt=prompt if self.verbose else None,
            metadata=metadata or {}
        )

        # Emit start event
        if self.emitter:
            try:
                from agentic.events import SearchEvent, EventType
                await self.emitter.emit(SearchEvent(
                    event_type=EventType.LLM_CALL_START,
                    request_id=self.request_id,
                    data={
                        "agent": agent_name,
                        "operation": operation,
                        "model": model,
                        "prompt_tokens": call.prompt_length_tokens
                    }
                ))
            except Exception as e:
                logger.debug(f"Failed to emit LLM start event: {e}")

        try:
            yield call
        except Exception as e:
            call.finalize(error=str(e), parse_success=False)
            raise
        finally:
            # Ensure call is finalized
            if call.end_time is None:
                call.finalize()

            # Store verbose response
            if self.verbose and hasattr(call, '_response_for_logging'):
                call.full_response = call._response_for_logging

            # Update aggregates
            self._total_input_tokens += call.prompt_length_tokens
            self._total_output_tokens += call.response_length_tokens
            self._total_latency_ms += call.latency_ms
            self._calls_by_agent[agent_name] = self._calls_by_agent.get(agent_name, 0) + 1
            self._calls_by_operation[operation] = self._calls_by_operation.get(operation, 0) + 1

            # Log summary
            status = "✓" if call.parse_success and not call.error_message else "✗"
            logger.info(
                f"[{self.request_id}] LLM {status}: {agent_name}.{operation} | "
                f"model={model} | in={call.prompt_length_tokens}tok | "
                f"out={call.response_length_tokens}tok | {call.latency_ms}ms"
            )

            if self.verbose:
                logger.debug(f"[{self.request_id}] LLM PROMPT:\n{prompt[:1000]}...")
                if call.full_response:
                    logger.debug(f"[{self.request_id}] LLM RESPONSE:\n{call.full_response[:1000]}...")

            # Store call
            self.calls.append(call)

            # Emit complete event
            if self.emitter:
                try:
                    from agentic.events import SearchEvent, EventType
                    await self.emitter.emit(SearchEvent(
                        event_type=EventType.LLM_CALL_COMPLETE,
                        request_id=self.request_id,
                        data=call.to_summary_dict()
                    ))
                except Exception as e:
                    logger.debug(f"Failed to emit LLM complete event: {e}")

    def track_call_sync(
        self,
        agent_name: str,
        operation: str,
        model: str,
        prompt: str,
        prompt_template: str = "custom",
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMCall:
        """
        Synchronous version - create a call record manually.

        Returns LLMCall that must be finalized after the call completes.
        """
        context_items = (
            prompt.count("[Source") +
            prompt.count("Finding:") +
            prompt.count("Question:")
        )

        call = LLMCall(
            request_id=self.request_id,
            agent_name=agent_name,
            operation=operation,
            model=model,
            prompt_template=prompt_template,
            prompt_length_chars=len(prompt),
            prompt_length_tokens=len(prompt) // 4,
            input_context_items=context_items,
            full_prompt=prompt if self.verbose else None,
            metadata=metadata or {}
        )

        return call

    def finalize_call_sync(self, call: LLMCall, response: str = "", parse_success: bool = True, error: str = None):
        """Finalize a synchronously-created call."""
        call.finalize(response=response, parse_success=parse_success, error=error)

        # Update aggregates
        self._total_input_tokens += call.prompt_length_tokens
        self._total_output_tokens += call.response_length_tokens
        self._total_latency_ms += call.latency_ms
        self._calls_by_agent[call.agent_name] = self._calls_by_agent.get(call.agent_name, 0) + 1
        self._calls_by_operation[call.operation] = self._calls_by_operation.get(call.operation, 0) + 1

        # Log
        status = "✓" if parse_success and not error else "✗"
        logger.info(
            f"[{self.request_id}] LLM {status}: {call.agent_name}.{call.operation} | "
            f"model={call.model} | in={call.prompt_length_tokens}tok | "
            f"out={call.response_length_tokens}tok | {call.latency_ms}ms"
        )

        self.calls.append(call)

    def get_call_summary(self) -> Dict[str, Any]:
        """
        Generate summary of all LLM calls for this request.

        Returns:
            Summary dict suitable for response metadata
        """
        if not self.calls:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_latency_ms": 0,
                "by_agent": {},
                "by_operation": {}
            }

        successful_calls = [c for c in self.calls if c.parse_success and not c.error_message]
        failed_calls = [c for c in self.calls if not c.parse_success or c.error_message]

        return {
            "total_calls": len(self.calls),
            "successful_calls": len(successful_calls),
            "failed_calls": len(failed_calls),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": self._total_latency_ms // len(self.calls) if self.calls else 0,
            "by_agent": dict(self._calls_by_agent),
            "by_operation": dict(self._calls_by_operation),
            "call_chain": [c.to_summary_dict() for c in self.calls[:20]]  # Limit for response size
        }

    def get_calls_by_agent(self, agent_name: str) -> List[LLMCall]:
        """Get all calls made by a specific agent."""
        return [c for c in self.calls if c.agent_name == agent_name]

    def get_calls_by_operation(self, operation: str) -> List[LLMCall]:
        """Get all calls of a specific operation type."""
        return [c for c in self.calls if c.operation == operation]

    def get_slowest_calls(self, n: int = 5) -> List[LLMCall]:
        """Get the N slowest LLM calls."""
        return sorted(self.calls, key=lambda c: c.latency_ms, reverse=True)[:n]

    def get_failed_calls(self) -> List[LLMCall]:
        """Get all failed LLM calls."""
        return [c for c in self.calls if not c.parse_success or c.error_message]

    def export_for_debugging(self) -> List[Dict[str, Any]]:
        """Export all calls in full detail for debugging."""
        return [c.to_log_dict() for c in self.calls]


def get_llm_logger(
    request_id: str,
    emitter: Optional[Any] = None,
    verbose: bool = False
) -> LLMCallLogger:
    """
    Factory function to get an LLMCallLogger instance.

    Args:
        request_id: Unique request identifier
        emitter: Optional SSE event emitter
        verbose: Enable verbose logging (captures full prompts/responses)

    Returns:
        LLMCallLogger instance
    """
    return LLMCallLogger(request_id, emitter, verbose)
