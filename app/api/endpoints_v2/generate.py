import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.core.fastapi_depends import (
    ExamRagServiceDep,
    MindmapRagServiceDep,
    SlideRagServiceDep,
)
from app.schemas.exam_content import (
    ExamMatrix,
    GenerateMatrixRequest,
    GenerateQuestionsFromTopicRequest,
    Question,
)
from app.schemas.mindmap_content import MindmapGenerateRequest
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)
from app.schemas.token_usage import TokenUsage
from app.services.base_rag_service import ContentMismatchError
from app.utils.server_sent_event import sse_json_by_json, sse_word_by_word

logger = logging.getLogger(__name__)


class GenerateResponse(BaseModel):
    """Generic response wrapper with token usage."""

    data: Any
    token_usage: TokenUsage | None = None


router = APIRouter(tags=["generate"])


@router.post("/outline/generate")
def generate_outline_with_rag(
    outlineGenerateRequest: OutlineGenerateRequest, svc: SlideRagServiceDep
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


@router.post("/presentations/generate")
def generate_presentation_with_rag(
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: SlideRagServiceDep,
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
    mindmapGenerateRequest: MindmapGenerateRequest, svc: MindmapRagServiceDep
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
    svc: SlideRagServiceDep,
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
    svc: SlideRagServiceDep,
):
    try:
        chunks = svc.make_presentation_rag_stream(presentationGenerateRequest)
    except ContentMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return EventSourceResponse(sse_json_by_json(request, chunks), ping=None)


@router.post("/exams/matrix/generate", response_model=ExamMatrix)
def generate_exam_matrix_with_rag(
    request: GenerateMatrixRequest, svc: ExamRagServiceDep
):
    """
    Generate a 3D exam matrix based on topics and prerequisites using RAG.
    """
    logger.info(
        f"[EXAM/MATRIX/RAG/GENERATE] Received request for matrix: {request.name}"
    )

    try:
        result = svc.generate_matrix_with_rag(request)
        token_usage = svc.last_token_usage
        logger.info(
            f"[EXAM/MATRIX/RAG/GENERATE] Successfully generated matrix. "
            f"Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, "
            f"total={token_usage.total_tokens}, model={token_usage.model}"
        )
        return result
    except ContentMismatchError as e:
        logger.error(f"[EXAM/MATRIX/RAG/GENERATE] Content mismatch: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        logger.error(f"[EXAM/MATRIX/RAG/GENERATE] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[EXAM/MATRIX/RAG/GENERATE] Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate matrix: {str(e)}",
        )


@router.post("/questions/generate", response_model=list[Question])
def generate_questions_with_rag(
    request: GenerateQuestionsFromTopicRequest, svc: ExamRagServiceDep
):
    """
    Generate questions based on topic and requirements using RAG.

    This endpoint uses AI with RAG to create exam questions matching the Question entity schema.
    """
    logger.info(
        f"[QUESTIONS/RAG/GENERATE] Received request for topic: {request.topic}, grade: {request.grade}"
    )

    try:
        result = svc.generate_questions_with_rag(request)
        token_usage = svc.last_token_usage
        logger.info(
            f"[QUESTIONS/RAG/GENERATE] Successfully generated {len(result)} questions. "
            f"Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, "
            f"total={token_usage.total_tokens}, model={token_usage.model}"
        )
        return result
    except ContentMismatchError as e:
        logger.error(f"[QUESTIONS/RAG/GENERATE] Content mismatch: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        logger.error(f"[QUESTIONS/RAG/GENERATE] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"[QUESTIONS/RAG/GENERATE] File not found: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prompt template not found: {str(e)}",
        )
    except Exception as e:
        logger.error(f"[QUESTIONS/RAG/GENERATE] Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate questions: {str(e)}",
        )
