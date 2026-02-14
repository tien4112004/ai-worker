"""API endpoints for exam and question generation."""

import json
from typing import List

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.core.fastapi_depends import ExamServiceDep
from app.schemas.exam_content import (
    ExamMatrix,
    GenerateMatrixRequest,
    GenerateQuestionsFromMatrixRequest,
    GenerateQuestionsFromMatrixResponse,
    GenerateQuestionsRequest,
    MatrixItem,
    Question,
)

router = APIRouter(tags=["exams"])


@router.post("/exams/generate-matrix", response_model=ExamMatrix)
def generate_exam_matrix(
    request_body: GenerateMatrixRequest, svc: ExamServiceDep
):
    """
    Generate a 3D exam matrix based on topics, difficulties, and question types.

    This endpoint creates a structured 3D blueprint for an exam, with dimensions:
    - Topics (first dimension)
    - Difficulties: KNOWLEDGE, COMPREHENSION, APPLICATION (second dimension)
    - Question types: MULTIPLE_CHOICE, FILL_IN_BLANK, etc. (third dimension)

    Each cell contains {count, points} representing the number of questions
    and total points for that combination.
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


# Question Generation Endpoints
@router.post(
    "/exams/generate-questions-from-matrix",
    response_class=JSONResponse,
)
def generate_questions_from_matrix(
    request_body: GenerateQuestionsFromMatrixRequest, svc: ExamServiceDep
):
    """
    Generate questions from matrix - returns raw LLM JSON response.

    The backend handles all parsing, validation, and data enrichment.

    Supports:
    - Context-based questions (contexts pre-selected by backend)
    - Regular curriculum questions
    - Batch generation in single LLM call

    Returns:
        Raw JSON string with questions array from LLM
    """
    try:
        # Service returns raw JSON string
        raw_json = svc.generate_questions_from_matrix(request_body)

        # Parse to validate it's valid JSON, then return as-is
        import json

        parsed = json.loads(raw_json)
        return JSONResponse(content=parsed)
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
