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
    - Difficulties: easy, medium, hard (second dimension)
    - Question types: multiple_choice, fill_in_blank, etc. (third dimension)

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
    "/exams/generate-questions-from-matrix", response_model=List[Question]
)
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
