"""API endpoints for exam and question generation."""

import json
from typing import List

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from app.depends import ExamServiceDep
from app.schemas.exam_content import (
    ExamMatrix,
    GenerateMatrixRequest,
    GenerateQuestionsRequest,
    MatrixItem,
    QuestionWithContext,
)

router = APIRouter(tags=["exams"])


# Matrix Generation Endpoints
@router.post("/exams/generate-matrix", response_model=ExamMatrix)
def generate_exam_matrix(
    request_body: GenerateMatrixRequest, svc: ExamServiceDep
):
    """
    Generate an exam matrix based on topic and requirements.

    This endpoint creates a structured blueprint for an exam, specifying
    what questions should be generated, their types, and requirements.
    """
    try:
        result = svc.generate_matrix(request_body)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate matrix: {str(e)}",
        )


@router.post("/exams/generate-matrix/stream")
async def generate_exam_matrix_stream(
    request: Request,
    request_body: GenerateMatrixRequest,
    svc: ExamServiceDep,
):
    """
    Generate an exam matrix with streaming response (Server-Sent Events).

    Provides real-time feedback as the matrix is being generated.
    """
    try:
        result = svc.generate_matrix_stream(request_body)

        async def event_stream():
            async for chunk in result:
                if await request.is_disconnected():
                    break
                yield {"data": chunk}

        return EventSourceResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate matrix stream: {str(e)}",
        )


# Question Generation Endpoints (Priority 1)
@router.post("/exams/generate-questions", response_model=List[QuestionWithContext])
def generate_questions(
    request_body: GenerateQuestionsRequest, svc: ExamServiceDep
):
    """
    Generate questions based on an exam matrix.

    This is the high-priority endpoint that takes an approved matrix
    and generates actual exam questions with contexts, answers, and explanations.
    """
    try:
        result = svc.generate_questions_from_matrix(request_body)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate questions: {str(e)}",
        )


@router.post("/exams/generate-questions/stream")
async def generate_questions_stream(
    request: Request,
    request_body: GenerateQuestionsRequest,
    svc: ExamServiceDep,
):
    """
    Generate questions from matrix with streaming progress updates (SSE).

    Provides real-time progress updates and the final generated questions.
    Returns status updates with progress information and the completed questions.
    """
    try:
        result = svc.generate_questions_from_matrix_stream(request_body)

        async def event_stream():
            async for data in result:
                if await request.is_disconnected():
                    break
                # Send as JSON event
                yield {"data": json.dumps(data, ensure_ascii=False)}

        return EventSourceResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate questions stream: {str(e)}",
        )


# Mock Endpoints for Testing
@router.post("/exams/generate-matrix/mock", response_model=ExamMatrix)
def generate_exam_matrix_mock(
    request_body: GenerateMatrixRequest, svc: ExamServiceDep
):
    """Generate a mock exam matrix for testing without LLM calls."""
    result = svc.generate_matrix_mock(request_body)
    return result


@router.post("/exams/generate-matrix/stream/mock")
async def generate_exam_matrix_stream_mock(
    request: Request,
    request_body: GenerateMatrixRequest,
    svc: ExamServiceDep,
):
    """Generate a mock matrix stream for testing without LLM calls."""

    async def event_stream():
        async for chunk in svc.generate_matrix_stream_mock(request_body):
            if await request.is_disconnected():
                break
            yield {"data": chunk}

    return EventSourceResponse(event_stream(), media_type="text/event-stream")


@router.post("/exams/generate-questions/mock", response_model=List[QuestionWithContext])
def generate_questions_mock(
    request_body: GenerateQuestionsRequest, svc: ExamServiceDep
):
    """Generate mock questions for testing without LLM calls."""
    result = svc.generate_questions_mock(request_body)
    return result


@router.post("/exams/generate-questions/stream/mock")
async def generate_questions_stream_mock(
    request: Request,
    request_body: GenerateQuestionsRequest,
    svc: ExamServiceDep,
):
    """Generate mock questions stream for testing without LLM calls."""

    async def event_stream():
        async for data in svc.generate_questions_stream_mock(request_body):
            if await request.is_disconnected():
                break
            yield {"data": json.dumps(data, ensure_ascii=False)}

    return EventSourceResponse(event_stream(), media_type="text/event-stream")
