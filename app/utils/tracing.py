from contextlib import contextmanager

from openinference.instrumentation import using_attributes

from app.utils.constants import RAG_STATUS_NO_RAG


@contextmanager
def trace_generation(
    request,
    body,
    resource_type: str,
    generation_type: str,
    rag_status: str = RAG_STATUS_NO_RAG,
):
    """Context manager to set trace attributes for auto-instrumentation using Phoenix semantic conventions.

    This uses the OpenInference context manager approach which ensures attributes are properly
    propagated to all spans created by auto-instrumentation (e.g., LangChain spans).

    Args:
        request: FastAPI Request object
        body: Request body (Pydantic model)
        resource_type: Type of resource being generated (presentation, exam, mindmap, image)
        generation_type: Generation mode (batch, stream)
        rag_status: RAG usage status (rag, no-rag) - defaults to no-rag

    Sets in OpenTelemetry Context:
        - user.id: User identifier from request header
        - metadata: JSON dict with grade, subject, resource_type, generation_type, rag_status
        - tag.tags: List of categorization tags [rag_status, resource_type, generation_type]

    Usage:
        with trace_generation(request, body, RESOURCE_TYPE_PRESENTATION, GENERATION_TYPE_BATCH):
            # All spans created here will inherit the attributes
            result = svc.make_outline(outlineGenerateRequest)
    """
    # Extract user ID from request headers
    user_id = request.headers.get("X-User-ID")

    print("[INFO] Tracing generation - User ID:", user_id)

    # Extract grade and subject from request body
    grade = getattr(body, "grade", None) or getattr(body, "gradeLevel", None)
    subject = getattr(body, "subject", None)

    # Build metadata dict following OpenInference semantic conventions
    metadata = {
        "resource_type": resource_type,
        "generation_type": generation_type,
        "rag_status": rag_status,
    }
    if grade:
        metadata["grade"] = grade
    if subject:
        metadata["subject"] = subject

    # Build tags list
    tags = [rag_status, resource_type, generation_type]

    # Use OpenInference context manager to set attributes
    # This ensures auto-instrumentation picks up these attributes
    with using_attributes(
        user_id=user_id if user_id else 'anonymous',
        metadata=metadata,
        tags=tags,
    ):
        yield