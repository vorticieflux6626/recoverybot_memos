"""
OpenTelemetry Distributed Tracing for Agentic Search Pipeline.

Part of G.1.3: Add OpenTelemetry basic tracing.

Provides distributed tracing across the agentic search pipeline:
- Span creation for key operations (search, synthesis, verification)
- Context propagation across async boundaries
- Integration with OTLP exporters (Jaeger, Zipkin, etc.)
- Convenience decorators for automatic span creation

Usage:
    from agentic.tracing import (
        get_tracer,
        trace_operation,
        TracingMiddleware,
        configure_tracing
    )

    # Configure tracing at startup
    configure_tracing(service_name="memos-agentic", endpoint="http://localhost:4318")

    # Trace a function
    @trace_operation("search_web")
    async def search_web(query: str):
        ...

    # Manual span creation
    with get_tracer().start_as_current_span("custom_operation") as span:
        span.set_attribute("query", query)
        ...

Research Basis:
- OpenTelemetry Specification v1.0
- Distributed Tracing Best Practices (W3C Trace Context)
- OTLP Protocol (gRPC and HTTP)
"""

import logging
import functools
import asyncio
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from contextlib import contextmanager
from datetime import datetime, timezone

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, Span
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.context import Context

    # Optional OTLP exporter
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        OTLP_AVAILABLE = True
    except ImportError:
        OTLP_AVAILABLE = False
        OTLPSpanExporter = None

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    TracerProvider = None
    Span = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None
    OTLPSpanExporter = None
    Resource = None
    SERVICE_NAME = None
    Status = None
    StatusCode = None
    TraceContextTextMapPropagator = None
    Context = None
    OTLP_AVAILABLE = False

logger = logging.getLogger("agentic.tracing")

# Type variable for generic function decoration
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# GenAI Semantic Conventions (OpenTelemetry 2025)
# Based on: https://opentelemetry.io/docs/specs/semconv/gen-ai/
# =============================================================================
class GenAIAttributes:
    """OpenTelemetry GenAI semantic convention attribute names."""
    # System attributes
    SYSTEM = "gen_ai.system"  # e.g., "ollama", "openai"

    # Request attributes
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"

    # Response attributes
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"  # Actual model used
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Usage attributes
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # Operation attributes (custom extension)
    OPERATION_NAME = "gen_ai.operation.name"  # analysis, synthesis, etc.
    AGENT_ROLE = "gen_ai.agent.role"  # analyzer, synthesizer, verifier
    PROMPT_TEMPLATE = "gen_ai.prompt.template"  # Template name used

# Global configuration
_tracer_provider: Optional["TracerProvider"] = None
_tracer: Optional[Any] = None
_tracing_enabled: bool = False
_service_name: str = "memos-agentic"


class TracingConfig:
    """Configuration for OpenTelemetry tracing."""

    def __init__(
        self,
        service_name: str = "memos-agentic",
        service_version: str = "0.44.0",
        environment: str = "development",
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
        enabled: bool = True,
        sample_rate: float = 1.0,
    ):
        """
        Initialize tracing configuration.

        Args:
            service_name: Name of the service for traces
            service_version: Version of the service
            environment: Environment (development, staging, production)
            otlp_endpoint: OTLP HTTP endpoint (e.g., http://localhost:4318/v1/traces)
            console_export: Whether to export spans to console (for debugging)
            enabled: Whether tracing is enabled
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.otlp_endpoint = otlp_endpoint
        self.console_export = console_export
        self.enabled = enabled
        self.sample_rate = sample_rate


def configure_tracing(config: Optional[TracingConfig] = None, **kwargs) -> bool:
    """
    Configure OpenTelemetry tracing for the agentic search pipeline.

    Args:
        config: TracingConfig instance or None to use kwargs
        **kwargs: Passed to TracingConfig if config is None

    Returns:
        True if tracing was configured successfully
    """
    global _tracer_provider, _tracer, _tracing_enabled, _service_name

    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - tracing disabled")
        return False

    if config is None:
        config = TracingConfig(**kwargs)

    if not config.enabled:
        logger.info("Tracing disabled by configuration")
        return False

    _service_name = config.service_name

    try:
        # Create resource with service info
        resource = Resource.create({
            SERVICE_NAME: config.service_name,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
            "service.namespace": "memos",
        })

        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)

        # Add exporters
        if config.console_export:
            console_exporter = ConsoleSpanExporter()
            _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
            logger.info("Console span exporter enabled")

        if config.otlp_endpoint and OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
            _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP exporter configured: {config.otlp_endpoint}")
        elif config.otlp_endpoint and not OTLP_AVAILABLE:
            logger.warning("OTLP endpoint specified but exporter not installed")

        # Set global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Get tracer
        _tracer = trace.get_tracer(
            config.service_name,
            config.service_version
        )

        _tracing_enabled = True
        logger.info(
            f"OpenTelemetry tracing configured: service={config.service_name}, "
            f"version={config.service_version}, env={config.environment}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to configure tracing: {e}")
        _tracing_enabled = False
        return False


def get_tracer() -> Any:
    """
    Get the configured tracer instance.

    Returns:
        OpenTelemetry Tracer or NoOpTracer if tracing is disabled
    """
    global _tracer

    if not OTEL_AVAILABLE:
        return NoOpTracer()

    if _tracer is None:
        # Auto-configure with defaults if not configured
        if not _tracing_enabled:
            configure_tracing(enabled=True, console_export=False)

        if _tracer is None:
            _tracer = trace.get_tracer(_service_name)

    return _tracer


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return _tracing_enabled and OTEL_AVAILABLE


class NoOpSpan:
    """No-operation span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def set_status(self, status: Any, description: Optional[str] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


class NoOpTracer:
    """No-operation tracer for when OpenTelemetry is not available."""

    def start_as_current_span(
        self,
        name: str,
        **kwargs
    ) -> NoOpSpan:
        return NoOpSpan()

    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()


def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True
) -> Callable[[F], F]:
    """
    Decorator to trace a function or coroutine.

    Args:
        operation_name: Name of the operation for the span
        attributes: Static attributes to add to the span
        record_exception: Whether to record exceptions in the span

    Example:
        @trace_operation("search_web", {"component": "searcher"})
        async def search_web(query: str):
            ...
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(operation_name) as span:
                    if attributes:
                        _set_span_attributes(span, attributes)

                    # Extract common attributes from kwargs
                    if "request_id" in kwargs:
                        _set_span_attribute(span, "request_id", kwargs["request_id"])
                    if "query" in kwargs:
                        _set_span_attribute(span, "query", str(kwargs["query"])[:200])

                    try:
                        result = await func(*args, **kwargs)
                        _set_span_status(span, StatusCode.OK if OTEL_AVAILABLE else None)
                        return result
                    except Exception as e:
                        if record_exception:
                            _record_exception(span, e)
                        _set_span_status(span, StatusCode.ERROR if OTEL_AVAILABLE else None, str(e))
                        raise
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(operation_name) as span:
                    if attributes:
                        _set_span_attributes(span, attributes)

                    try:
                        result = func(*args, **kwargs)
                        _set_span_status(span, StatusCode.OK if OTEL_AVAILABLE else None)
                        return result
                    except Exception as e:
                        if record_exception:
                            _record_exception(span, e)
                        _set_span_status(span, StatusCode.ERROR if OTEL_AVAILABLE else None, str(e))
                        raise
            return sync_wrapper  # type: ignore
    return decorator


# Helper functions for span operations
def _set_span_attribute(span: Any, key: str, value: Any) -> None:
    """Set a span attribute safely."""
    if hasattr(span, 'set_attribute'):
        try:
            # Convert value to a supported type
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
            elif isinstance(value, (list, tuple)):
                span.set_attribute(key, str(value)[:500])
            else:
                span.set_attribute(key, str(value)[:500])
        except Exception:
            pass


def _set_span_attributes(span: Any, attributes: Dict[str, Any]) -> None:
    """Set multiple span attributes safely."""
    for key, value in attributes.items():
        _set_span_attribute(span, key, value)


def _set_span_status(span: Any, status_code: Any, description: Optional[str] = None) -> None:
    """Set span status safely."""
    if OTEL_AVAILABLE and hasattr(span, 'set_status') and status_code is not None:
        try:
            span.set_status(Status(status_code, description))
        except Exception:
            pass


def _record_exception(span: Any, exception: Exception) -> None:
    """Record exception in span safely."""
    if hasattr(span, 'record_exception'):
        try:
            span.record_exception(exception)
        except Exception:
            pass


def _add_span_event(span: Any, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Add an event to a span safely."""
    if hasattr(span, 'add_event'):
        try:
            span.add_event(name, attributes or {})
        except Exception:
            pass


# Span context for manual tracing
@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[Any] = None
):
    """
    Context manager for creating a traced span.

    Args:
        name: Name of the span
        attributes: Attributes to add to the span
        kind: SpanKind (optional)

    Example:
        with trace_span("process_documents", {"count": 10}) as span:
            span.add_event("started_processing")
            ...
    """
    tracer = get_tracer()
    kwargs = {}
    if kind is not None:
        kwargs['kind'] = kind

    with tracer.start_as_current_span(name, **kwargs) as span:
        if attributes:
            _set_span_attributes(span, attributes)
        try:
            yield span
        except Exception as e:
            _record_exception(span, e)
            _set_span_status(span, StatusCode.ERROR if OTEL_AVAILABLE else None, str(e))
            raise


# Agentic-specific tracing helpers
class AgenticTracer:
    """
    High-level tracing helper for agentic search pipeline.

    Provides semantic span creation for common agentic operations.
    """

    def __init__(self, request_id: str):
        """
        Initialize agentic tracer for a request.

        Args:
            request_id: Unique request identifier
        """
        self.request_id = request_id
        self.tracer = get_tracer()
        self._root_span: Optional[Any] = None

    @contextmanager
    def trace_search_pipeline(
        self,
        query: str,
        preset: str = "balanced"
    ):
        """Trace the entire search pipeline."""
        with self.tracer.start_as_current_span("agentic_search_pipeline") as span:
            self._root_span = span
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "query": query[:200],
                "preset": preset,
                "pipeline.type": "agentic",
            })
            try:
                yield span
            except Exception as e:
                _record_exception(span, e)
                raise

    @contextmanager
    def trace_analysis(self, query: str):
        """Trace query analysis phase."""
        with self.tracer.start_as_current_span("analyze_query") as span:
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "query": query[:200],
                "phase": "analysis",
            })
            yield span

    @contextmanager
    def trace_planning(self):
        """Trace search planning phase."""
        with self.tracer.start_as_current_span("plan_search") as span:
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "phase": "planning",
            })
            yield span

    @contextmanager
    def trace_search(self, iteration: int = 1, query_count: int = 1):
        """Trace web search phase."""
        with self.tracer.start_as_current_span("web_search") as span:
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "phase": "search",
                "iteration": iteration,
                "query_count": query_count,
            })
            yield span

    @contextmanager
    def trace_scraping(self, url_count: int = 1):
        """Trace URL scraping phase."""
        with self.tracer.start_as_current_span("scrape_urls") as span:
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "phase": "scraping",
                "url_count": url_count,
            })
            yield span

    @contextmanager
    def trace_crag_evaluation(self, document_count: int = 0):
        """Trace CRAG retrieval evaluation."""
        with self.tracer.start_as_current_span("crag_evaluation") as span:
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "phase": "crag_evaluation",
                "document_count": document_count,
            })
            yield span

    @contextmanager
    def trace_synthesis(self, source_count: int = 0):
        """Trace answer synthesis phase."""
        with self.tracer.start_as_current_span("synthesize_answer") as span:
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "phase": "synthesis",
                "source_count": source_count,
            })
            yield span

    @contextmanager
    def trace_verification(self, claim_count: int = 0):
        """Trace claim verification phase."""
        with self.tracer.start_as_current_span("verify_claims") as span:
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "phase": "verification",
                "claim_count": claim_count,
            })
            yield span

    @contextmanager
    def trace_self_reflection(self):
        """Trace self-RAG reflection phase."""
        with self.tracer.start_as_current_span("self_rag_reflection") as span:
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "phase": "self_reflection",
            })
            yield span

    @contextmanager
    def trace_llm_call(
        self,
        model: str,
        operation: str,
        input_tokens: int = 0
    ):
        """Trace an LLM API call (legacy)."""
        with self.tracer.start_as_current_span(f"llm_{operation}") as span:
            _set_span_attributes(span, {
                "request_id": self.request_id,
                "llm.model": model,
                "llm.operation": operation,
                "llm.input_tokens": input_tokens,
                "phase": "llm_call",
            })
            yield span

    @contextmanager
    def trace_llm_call_genai(
        self,
        model: str,
        operation: str,
        agent_role: str,
        input_tokens: int = 0,
        prompt_template: str = "custom",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: str = "ollama"
    ):
        """
        Trace an LLM API call with GenAI semantic conventions.

        Args:
            model: Model name (e.g., "qwen3:8b")
            operation: Operation type (analysis, synthesis, verification, etc.)
            agent_role: Agent making the call (analyzer, synthesizer, etc.)
            input_tokens: Estimated input token count
            prompt_template: Name of the prompt template used
            temperature: Temperature parameter if set
            max_tokens: Max tokens parameter if set
            system: LLM system (ollama, openai, etc.)

        Yields:
            Span with GenAI attributes that can be updated with response info
        """
        with self.tracer.start_as_current_span(f"gen_ai.{operation}") as span:
            # Set GenAI semantic convention attributes
            _set_span_attributes(span, {
                # Request context
                "request_id": self.request_id,
                # GenAI system
                GenAIAttributes.SYSTEM: system,
                GenAIAttributes.REQUEST_MODEL: model,
                GenAIAttributes.OPERATION_NAME: operation,
                GenAIAttributes.AGENT_ROLE: agent_role,
                GenAIAttributes.PROMPT_TEMPLATE: prompt_template,
                # Usage (input)
                GenAIAttributes.USAGE_INPUT_TOKENS: input_tokens,
            })

            # Optional parameters
            if temperature is not None:
                _set_span_attribute(span, GenAIAttributes.REQUEST_TEMPERATURE, temperature)
            if max_tokens is not None:
                _set_span_attribute(span, GenAIAttributes.REQUEST_MAX_TOKENS, max_tokens)

            try:
                yield span
            except Exception as e:
                _record_exception(span, e)
                _set_span_status(span, StatusCode.ERROR if OTEL_AVAILABLE else None, str(e))
                raise

    def update_llm_response(
        self,
        span: Any,
        output_tokens: int,
        response_model: Optional[str] = None,
        finish_reason: str = "stop"
    ):
        """
        Update span with LLM response information.

        Call this after receiving the LLM response within the trace_llm_call_genai context.

        Args:
            span: The span from trace_llm_call_genai
            output_tokens: Number of output tokens
            response_model: Actual model used (if different from requested)
            finish_reason: Finish reason (stop, length, etc.)
        """
        _set_span_attributes(span, {
            GenAIAttributes.USAGE_OUTPUT_TOKENS: output_tokens,
            GenAIAttributes.RESPONSE_FINISH_REASONS: finish_reason,
        })
        if response_model:
            _set_span_attribute(span, GenAIAttributes.RESPONSE_MODEL, response_model)

        # Calculate total tokens
        input_tokens = 0
        try:
            # Try to get input tokens from existing span attributes
            if hasattr(span, 'attributes'):
                input_tokens = span.attributes.get(GenAIAttributes.USAGE_INPUT_TOKENS, 0)
        except Exception:
            pass
        _set_span_attribute(span, GenAIAttributes.USAGE_TOTAL_TOKENS, input_tokens + output_tokens)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the root span."""
        if self._root_span:
            _add_span_event(self._root_span, name, attributes)

    def set_result(
        self,
        success: bool,
        confidence: float = 0.0,
        source_count: int = 0,
        execution_time_ms: float = 0.0
    ):
        """Set final result attributes on the root span."""
        if self._root_span:
            _set_span_attributes(self._root_span, {
                "result.success": success,
                "result.confidence": round(confidence, 3),
                "result.source_count": source_count,
                "result.execution_time_ms": round(execution_time_ms, 1),
            })


def get_agentic_tracer(request_id: str) -> AgenticTracer:
    """
    Get an AgenticTracer for a request.

    Args:
        request_id: Unique request identifier

    Returns:
        AgenticTracer instance
    """
    return AgenticTracer(request_id)


# Export tracing status
def get_tracing_status() -> Dict[str, Any]:
    """Get current tracing configuration status."""
    return {
        "opentelemetry_available": OTEL_AVAILABLE,
        "otlp_available": OTLP_AVAILABLE,
        "tracing_enabled": _tracing_enabled,
        "service_name": _service_name,
        "tracer_configured": _tracer is not None,
    }


# Shutdown tracing
def shutdown_tracing():
    """Shutdown tracing and flush any pending spans."""
    global _tracer_provider, _tracer, _tracing_enabled

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("Tracing shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down tracing: {e}")

    _tracer_provider = None
    _tracer = None
    _tracing_enabled = False
