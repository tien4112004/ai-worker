import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.core.fastapi_depends import (
    ContentRagServiceDep,
    DocumentEmbeddingsRepositoryDep,
)
from app.schemas.mindmap_content import MindmapGenerateRequest
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)
from app.schemas.token_usage import TokenUsage
from app.services.content_rag_service import ContentMismatchError
from app.utils.server_sent_event import sse_json_by_json, sse_word_by_word

logger = logging.getLogger(__name__)


class GenerateResponse(BaseModel):
    """Generic response wrapper with token usage."""

    data: Any
    token_usage: TokenUsage | None = None


router = APIRouter(tags=["generate"])


@router.post("/outline/generate")
def generate_outline_with_rag(
    outlineGenerateRequest: OutlineGenerateRequest, svc: ContentRagServiceDep
):
    try:
        result = svc.make_outline_with_rag(outlineGenerateRequest)
    except ContentMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token_usage = svc.last_token_usage
    logger.info(
        f"[OUTLINE/RAG/GENERATE] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}"
    )
    return GenerateResponse(data=result, token_usage=token_usage)


class QueryRequest(BaseModel):
    query: str
    grade: Optional[int] = None
    subject_code: Optional[str] = None


@router.post("/query")
def rag_query(
    request: QueryRequest,
    svc: DocumentEmbeddingsRepositoryDep,
):
    filters = {}
    if request.grade is not None:
        filters["grade"] = request.grade
    if request.subject_code is not None:
        filters["subject_code"] = request.subject_code

    result = svc.similarity_search_with_score(
        request.query, filter=filters or None, k=5
    )

    return {
        "results": [
            {"document": doc.page_content, "score": score}
            for doc, score in result
        ]
    }


@router.post("/presentations/generate")
def generate_presentation_with_rag(
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentRagServiceDep,
):
    try:
        result = svc.make_presentation_with_rag(presentationGenerateRequest)
    except ContentMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token_usage = svc.last_token_usage
    logger.info(
        f"[PRESENTATIONS/RAG/GENERATE] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}"
    )
    return GenerateResponse(data=result, token_usage=token_usage)


@router.post("/mindmap/generate")
def generate_mindmap_with_rag(
    mindmapGenerateRequest: MindmapGenerateRequest, svc: ContentRagServiceDep
):
    try:
        result = svc.generate_mindmap_with_rag(mindmapGenerateRequest)
    except ContentMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token_usage = svc.last_token_usage
    logger.info(
        f"[MINDMAP/RAG/GENERATE] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}"
    )
    return GenerateResponse(data=result, token_usage=token_usage)


@router.post("/outline/generate/stream")
def generate_outline_rag_stream(
    request: Request,
    outlineGenerateRequest: OutlineGenerateRequest,
    svc: ContentRagServiceDep,
):
    try:
        chunks = svc.make_outline_rag_stream(outlineGenerateRequest)
    except ContentMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return EventSourceResponse(sse_word_by_word(request, chunks), ping=None)


@router.post("/presentations/generate/stream")
def generate_presentation_rag_stream(
    request: Request,
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentRagServiceDep,
):
    try:
        chunks = svc.make_presentation_rag_stream(presentationGenerateRequest)
    except ContentMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return EventSourceResponse(sse_json_by_json(request, chunks), ping=None)


@router.post("/mindmap/generate/stream")
def generate_mindmap_rag_stream(
    request: Request,
    mindmapGenerateRequest: MindmapGenerateRequest,
    svc: ContentRagServiceDep,
):
    try:
        chunks = svc.generate_mindmap_rag_stream(mindmapGenerateRequest)
    except ContentMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return EventSourceResponse(sse_word_by_word(request, chunks), ping=None)
