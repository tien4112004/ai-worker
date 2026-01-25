from typing import Annotated

from fastapi import Depends, Request

from app.core.config import settings
from app.llms.executor import LLMExecutor
from app.services.content_service import ContentService
from app.services.exam_service import ExamService


def get_logger():
    return settings.logger


def get_content_service(request: Request) -> ContentService:
    """Get the content service with the default model."""
    return request.app.state.content_service


def get_exam_service(request: Request) -> ExamService:
    """Get the exam service."""
    return request.app.state.exam_service


ContentServiceDep = Annotated[ContentService, Depends(get_content_service)]
ExamServiceDep = Annotated[ExamService, Depends(get_exam_service)]
