from typing import Annotated

from fastapi import Depends, Request

from app.core.config import settings
from app.repositories.document_embeddings_repository import (
    DocumentEmbeddingsRepository,
)
from app.services.content_rag_service import ContentRagService
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


def get_doc_repository(request: Request):
    """Get the document embeddings repository."""
    return request.app.state.document_embeddings_repository


def get_content_rag_service(request: Request) -> ContentRagService:
    """Get the content rag service."""
    return request.app.state.content_rag_service


ExamServiceDep = Annotated[ExamService, Depends(get_exam_service)]
ContentRagServiceDep = Annotated[
    ContentRagService, Depends(get_content_rag_service)
]
ContentServiceDep = Annotated[ContentService, Depends(get_content_service)]
DocumentEmbeddingsRepositoryDep = Annotated[
    DocumentEmbeddingsRepository, Depends(get_doc_repository)
]
