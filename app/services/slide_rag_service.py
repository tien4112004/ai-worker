from typing import Generator

from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)
from app.services.base_rag_service import BaseRagService


class SlideRagService(BaseRagService):
    """Service for generating slide content (outlines and presentations) using RAG.

    Handles outline and presentation generation with document retrieval
    based on subject and grade filters.
    """

    def make_outline_with_rag(self, request: OutlineGenerateRequest) -> str:
        """Generate outline using LLM with RAG.

        Args:
            request: Request object containing parameters for outline generation

        Returns:
            Generated outline as a string

        Raises:
            ContentMismatchError: If retrieved documents don't match topic/subject/grade
        """
        sys_msg = self._system_with_subject_grade(
            "outline.system.rag",
            None,
            request.subject,
            request.grade,
        )

        usr_msg = self._system(
            "outline.user",
            request.to_dict(),
        )

        # Build filters for document search
        filters = self._build_filters(request.subject, request.grade)
        self._log_filters(filters, request.subject, request.grade)

        result, _ = self._rag_batch_call(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            filters=filters,
        )

        return result["answer"]

    def make_presentation_with_rag(
        self, request: PresentationGenerateRequest
    ) -> str:
        """Generate presentation using LLM with RAG.

        Args:
            request: Request object containing parameters for presentation generation

        Returns:
            Generated presentation as a string

        Raises:
            ContentMismatchError: If retrieved documents don't match topic/subject/grade
        """
        sys_msg = self._system_with_subject_grade(
            "presentation.system.rag",
            None,
            request.subject,
            request.grade,
        )

        usr_msg = self._system(
            "presentation.user",
            request.to_dict(),
        )

        filters = self._build_filters(request.subject, request.grade)
        self._log_filters(filters, request.subject, request.grade)

        result, _ = self._rag_batch_call(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            filters=filters,
        )

        return result["answer"]

    def make_outline_rag_stream(
        self, request: OutlineGenerateRequest
    ) -> Generator:
        """Generate outline using LLM with RAG in streaming mode.

        Args:
            request: Request object containing parameters for outline generation

        Returns:
            Generator yielding content chunks and final TokenUsage

        Raises:
            ContentMismatchError: If CONTENT_MISMATCH detected early in stream
        """
        sys_msg = self._system_with_subject_grade(
            "outline.system.rag", None, request.subject, request.grade
        )
        usr_msg = self._system("outline.user", request.to_dict())

        filters = self._build_filters(request.subject, request.grade)

        return self._checked_rag_stream(
            request.provider,
            request.model,
            usr_msg,
            sys_msg,
            filters if filters else None,
        )

    def make_presentation_rag_stream(
        self, request: PresentationGenerateRequest
    ) -> Generator:
        """Generate presentation using LLM with RAG in streaming mode.

        Args:
            request: Request object containing parameters for presentation generation

        Returns:
            Generator yielding content chunks and final TokenUsage

        Raises:
            ContentMismatchError: If CONTENT_MISMATCH detected early in stream
        """
        sys_msg = self._system_with_subject_grade(
            "presentation.system.rag", None, request.subject, request.grade
        )
        usr_msg = self._system("presentation.user", request.to_dict())

        filters = self._build_filters(request.subject, request.grade)

        return self._checked_rag_stream(
            request.provider,
            request.model,
            usr_msg,
            sys_msg,
            filters if filters else None,
        )
