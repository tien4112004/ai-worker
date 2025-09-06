
from typing import Annotated
from fastapi import Depends, Request
from app.llms.service import LLMService
from app.services.content_service import ContentService
from app.core.config import settings

def get_logger():
    return settings.logger

def get_content_service(request: Request) -> ContentService:
    """Get the content service with the default model."""
    model_name = request.app.state.settings.default_model
    get_logger().info(f"Using model: {model_name}")
    return request.app.state.content_service

ContentServiceDep = Annotated[ContentService, Depends(get_content_service)]

def get_llms_service(request: Request) -> LLMService:
    """Get the LLM service."""
    return request.app.state.llm_service

LLMServiceDep = Annotated[LLMService, Depends(get_llms_service)]