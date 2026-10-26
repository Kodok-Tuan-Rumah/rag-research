"""OpenTelemetry distributed tracing for RAG research API.

Provides:
- TracerProvider setup with OTLP HTTP exporter
- FastAPI instrumentation (automatic span creation)
- Helper functions for creating child spans
- Trace context propagation via W3C TraceContext
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("rag.tracing")

# Configuration from environment
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() in ("true", "1", "yes")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "rag-research")

_tracer = None
_tracer_provider = None


def init_tracing() -> bool:
    """Initialize OpenTelemetry tracing. Returns True if enabled."""
    global _tracer, _tracer_provider

    if not OTEL_ENABLED:
        logger.info("[TRACING] OpenTelemetry disabled (OTEL_ENABLED != true)")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{OTEL_ENDPOINT}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        # Use default composite propagator (includes TraceContext)

        _tracer_provider = provider
        _tracer = trace.get_tracer(OTEL_SERVICE_NAME)

        logger.info(f"[TRACING] OpenTelemetry enabled: service={OTEL_SERVICE_NAME}, endpoint={OTEL_ENDPOINT}")
        return True
    except Exception as e:
        logger.warning(f"[TRACING] Failed to initialize: {e}")
        return False


def get_tracer():
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer(OTEL_SERVICE_NAME)
        except Exception:
            pass
    return _tracer


def instrument_fastapi(app):
    """Instrument a FastAPI app with OpenTelemetry."""
    if not OTEL_ENABLED:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[TRACING] FastAPI instrumented")
    except Exception as e:
        logger.warning(f"[TRACING] FastAPI instrumentation failed: {e}")


def create_span(name: str, attributes: dict[str, Any] | None = None):
    """Create a child span context manager."""
    tracer = get_tracer()
    if tracer is None:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()

    from opentelemetry import trace
    span = tracer.start_as_current_span(name)
    if attributes:
        for k, v in attributes.items():
            span.set_attribute(k, v)
    return span


def get_trace_id() -> str:
    """Get the current trace ID from context."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return span.get_span_context().trace_id.to_bytes().hex()
    except Exception:
        pass
    return ""


def get_span_id() -> str:
    """Get the current span ID from context."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return span.get_span_context().span_id.to_bytes().hex()
    except Exception:
        pass
    return ""


def shutdown_tracing():
    """Shutdown the tracer provider."""
    global _tracer_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("[TRACING] Shutdown complete")
        except Exception as e:
            logger.warning(f"[TRACING] Shutdown failed: {e}")
        _tracer_provider = None
