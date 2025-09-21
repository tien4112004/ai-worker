from typing import Annotated

from fastapi import Depends, Request

from app.core.config import settings
from app.llms.executor import LLMExecutor
from app.services.content_service import ContentService


def get_logger():
    return settings.logger


def get_content_service(request: Request) -> ContentService:
    """Get the content service with the default model."""
    return request.app.state.content_service


ContentServiceDep = Annotated[ContentService, Depends(get_content_service)]
