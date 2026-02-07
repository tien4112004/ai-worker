"""Middleware to inject custom trace ID from request headers into OpenTelemetry context."""

import logging
import random

from fastapi import Request
from opentelemetry import context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

logger = logging.getLogger(__name__)


async def injectCustomTraceId(request: Request, call_next):
    """
    Inject custom trace ID from X-Trace-ID header into OpenTelemetry context.
    
    Supports trace ID formats:
    - UUID: "550e8400-e29b-41d4-a716-446655440000"
    - Hex string: "550e8400e29b41d4a716446655440000"
    - Shorter hex (will be zero-padded): "abc123"
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/handler in chain
        
    Returns:
        Response with X-Trace-ID header added
    """
    custom_trace_id = request.headers.get("X-Trace-ID")

    if custom_trace_id:
        try:
            logger.info(f"[TRACE_ID] Received X-Trace-ID header: {custom_trace_id}")
            
            # Remove spaces and hyphens for UUID format
            trace_id_hex = custom_trace_id.replace(" ", "").replace("-", "")

            # Ensure it's 32 hex characters (128 bits), pad if needed
            if len(trace_id_hex) < 32:
                trace_id_hex = trace_id_hex.zfill(32)
            elif len(trace_id_hex) > 32:
                trace_id_hex = trace_id_hex[:32]

            trace_id = int(trace_id_hex, 16)
            span_id = random.getrandbits(64)  # Generate random 64-bit span ID

            logger.info(f"[TRACE_ID] Converted to hex: {trace_id_hex}, span_id: {span_id}")

            # Create span context with custom trace ID
            span_context = SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=True,
                trace_flags=TraceFlags(0x01),  # Sampled
            )

            logger.info(f"[TRACE_ID] Created span context: trace_id={trace_id}, span_id={span_id}")

            # Set as parent context
            ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
            token = context.attach(ctx)

            try:
                # Create real span to export to Phoenix
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span("http_request") as span:
                    logger.info(f"[TRACE_ID] Created span with trace_id: {span.get_span_context().trace_id}")
                    span.set_attribute("http.method", request.method)
                    span.set_attribute("http.url", str(request.url))
                    span.set_attribute("http.trace_id", custom_trace_id)
                    span.set_attribute("http.path", request.url.path)
                    
                    logger.info(f"[TRACE_ID] Span attributes set for trace_id: {custom_trace_id}")
                    
                    response = await call_next(request)
                    # Add trace ID to response headers for debugging
                    response.headers["X-Trace-ID"] = custom_trace_id
                    span.set_attribute("http.status_code", response.status_code)
                    
                    logger.info(f"[TRACE_ID] Request completed with status {response.status_code} for trace_id: {custom_trace_id}")
                    return response
            except Exception as e:
                logger.error(f"[TRACE_ID] Error creating span for trace_id {custom_trace_id}: {str(e)}", exc_info=True)
                raise
            finally:
                context.detach(token)
        except (ValueError, TypeError) as e:
            # If trace ID parsing fails, continue without custom trace ID
            logger.error(f"[TRACE_ID] Invalid X-Trace-ID format '{custom_trace_id}': {str(e)}")

    return await call_next(request)
