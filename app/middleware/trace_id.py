"""Middleware to inject custom trace ID from request headers into OpenTelemetry context."""

import random

from fastapi import Request
from opentelemetry import context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

## trigger build again
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
            # Remove spaces and hyphens for UUID format
            trace_id_hex = custom_trace_id.replace(" ", "").replace("-", "")

            # Ensure it's 32 hex characters (128 bits), pad if needed
            if len(trace_id_hex) < 32:
                trace_id_hex = trace_id_hex.zfill(32)
            elif len(trace_id_hex) > 32:
                trace_id_hex = trace_id_hex[:32]

            trace_id = int(trace_id_hex, 16)
            span_id = random.getrandbits(64)  # Generate random 64-bit span ID

            # Create span context with custom trace ID
            span_context = SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=True,
                trace_flags=TraceFlags(0x01),  # Sampled
            )

            # Set as parent context
            ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
            token = context.attach(ctx)

            try:
                response = await call_next(request)
                # Add trace ID to response headers for debugging
                response.headers["X-Trace-ID"] = custom_trace_id
                return response
            finally:
                context.detach(token)
        except (ValueError, TypeError) as e:
            # If trace ID parsing fails, continue without custom trace ID
            print(f"Warning: Invalid X-Trace-ID format '{custom_trace_id}': {e}")

    return await call_next(request)
