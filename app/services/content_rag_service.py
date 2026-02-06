import warnings
from typing import Generator

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
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
from app.services.exam_rag_service import ExamRagService
from app.services.mindmap_rag_service import MindmapRagService
from app.services.slide_rag_service import SlideRagService


class ContentRagService:
    """Facade for backward compatibility. DEPRECATED.

    This class delegates all operations to the new specialized RAG services:
    - SlideRagService for outline and presentation generation
    - MindmapRagService for mindmap generation
    - ExamRagService for exam matrix and question generation

    Please use the specific services directly instead of this facade.
    This class will be removed in a future version.
    """

    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        warnings.warn(
            "ContentRagService is deprecated. Use SlideRagService, MindmapRagService, "
            "or ExamRagService instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._slide_service = SlideRagService(llm_executor, prompt_store)
        self._mindmap_service = MindmapRagService(llm_executor, prompt_store)
        self._exam_service = ExamRagService(llm_executor, prompt_store)

    @property
    def last_token_usage(self):
        """Aggregate token usage from all services.

        Returns the most recent token usage from any of the specialized services.
        """
        # Check each service and return the first non-None token usage
        for service in [
            self._slide_service,
            self._mindmap_service,
            self._exam_service,
        ]:
            if service.last_token_usage is not None:
                return service.last_token_usage
        return None

    def make_outline_with_rag(self, request: OutlineGenerateRequest) -> str:
        """Delegate to SlideRagService. DEPRECATED."""
        return self._slide_service.make_outline_with_rag(request)

    def make_presentation_with_rag(
        self, request: PresentationGenerateRequest
    ) -> str:
        """Delegate to SlideRagService. DEPRECATED."""
        return self._slide_service.make_presentation_with_rag(request)

    def make_outline_rag_stream(
        self, request: OutlineGenerateRequest
    ) -> Generator:
        """Delegate to SlideRagService. DEPRECATED."""
        return self._slide_service.make_outline_rag_stream(request)

    def make_presentation_rag_stream(
        self, request: PresentationGenerateRequest
    ) -> Generator:
        """Delegate to SlideRagService. DEPRECATED."""
        return self._slide_service.make_presentation_rag_stream(request)

    def generate_mindmap_with_rag(
        self, request: MindmapGenerateRequest
    ) -> str:
        """Delegate to MindmapRagService. DEPRECATED."""
        return self._mindmap_service.generate_mindmap_with_rag(request)

    def generate_mindmap_rag_stream(
        self, request: MindmapGenerateRequest
    ) -> Generator:
        """Delegate to MindmapRagService. DEPRECATED."""
        return self._mindmap_service.generate_mindmap_rag_stream(request)

    def generate_matrix_with_rag(
        self, request: GenerateMatrixRequest
    ) -> ExamMatrix:
        """Delegate to ExamRagService. DEPRECATED."""
        return self._exam_service.generate_matrix_with_rag(request)

    def generate_questions_with_rag(
        self, request: GenerateQuestionsFromTopicRequest
    ) -> list[Question]:
        """Delegate to ExamRagService. DEPRECATED."""
        return self._exam_service.generate_questions_with_rag(request)
