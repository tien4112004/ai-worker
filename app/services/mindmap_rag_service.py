from typing import Generator

from app.schemas.mindmap_content import MindmapGenerateRequest
from app.services.base_rag_service import BaseRagService


class MindmapRagService(BaseRagService):
    """Service for generating mindmaps using RAG.

    Handles mindmap generation with document retrieval
    based on subject and grade filters.
    """

    def generate_mindmap_with_rag(
        self, request: MindmapGenerateRequest
    ) -> str:
        """Generate mindmap using LLM with RAG.

        Args:
            request: Request object containing parameters for mindmap generation

        Returns:
            Generated mindmap as a string

        Raises:
            ContentMismatchError: If retrieved documents don't match topic/subject/grade
        """
        sys_msg = self._system_with_subject_grade(
            "mindmap.system.rag",
            request.to_dict(),
            request.subject,
            request.grade,
        )

        usr_msg = self._system(
            "mindmap.user",
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

    def generate_mindmap_rag_stream(
        self, request: MindmapGenerateRequest
    ) -> Generator:
        """Generate mindmap using LLM with RAG in streaming mode.

        Args:
            request: Request object containing parameters for mindmap generation

        Returns:
            Generator yielding content chunks and final TokenUsage

        Raises:
            ContentMismatchError: If CONTENT_MISMATCH detected early in stream
        """
        sys_msg = self._system_with_subject_grade(
            "mindmap.system.rag",
            request.to_dict(),
            request.subject,
            request.grade,
        )
        usr_msg = self._system("mindmap.user", request.to_dict())

        filters = self._build_filters(request.subject, request.grade)

        return self._checked_rag_stream(
            request.provider,
            request.model,
            usr_msg,
            sys_msg,
            filters if filters else None,
        )
