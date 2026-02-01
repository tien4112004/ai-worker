from typing import Any, Dict, List, Tuple

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.mindmap_content import MindmapGenerateRequest
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)


class ContentMismatchError(Exception):
    """Raised when retrieved documents do not match the requested topic/subject/grade."""

    pass


class ContentRagService:
    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()
        self.last_token_usage = None

    def _system(self, key: str, vars: Dict[str, Any] | None) -> str:
        return self.prompt_store.render(key, vars)

    @staticmethod
    def _check_content_mismatch(result: dict) -> None:
        answer = result.get("answer", "").strip()
        if answer.startswith("CONTENT_MISMATCH:"):
            message = answer[len("CONTENT_MISMATCH:") :].strip()
            raise ContentMismatchError(message)

    def make_outline_with_rag(self, request: OutlineGenerateRequest):
        """Generate outline using LLM and save result.
        Args:
            request (OutlineGenerateRequest): Request object containing parameters for outline generation.
        Returns:
            Dict: A dictionary containing the generated outline.
        """
        sys_msg = self._system(
            "outline.system.rag",
            None,
        )

        usr_msg = self._system(
            "outline.user",
            request.to_dict(),
        )

        # Build filters for document search
        # Note: Use subject_code (e.g., 'TV', 'T', 'TA') instead of subject name
        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            # Convert grade to integer if it's numeric (metadata stores it as int)
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        print(f"[DEBUG] RAG filters being applied: {filters}")
        print(
            f"[DEBUG] Request - subject: {request.subject}, grade: {request.grade} (type: {type(request.grade).__name__})"
        )

        result, token_usage = self.llm_executor.rag_batch(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            return_source_documents=True,
            filters=filters if filters else None,
            custom_prompt=None,
            verbose=False,
        )

        # Store token usage for later access
        self.last_token_usage = token_usage
        self._check_content_mismatch(result)
        return result["answer"]

    def make_presentation_with_rag(self, request: PresentationGenerateRequest):
        sys_msg = self._system(
            "presentation.system.rag",
            None,
        )

        usr_msg = self._system(
            "presentation.user",
            request.to_dict(),
        )

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        print(f"[DEBUG] RAG filters being applied: {filters}")
        print(
            f"[DEBUG] Request - subject: {request.subject}, grade: {request.grade} (type: {type(request.grade).__name__})"
        )

        result, token_usage = self.llm_executor.rag_batch(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            return_source_documents=True,
            filters=filters if filters else None,
            custom_prompt=None,
            verbose=False,
        )

        self.last_token_usage = token_usage
        self._check_content_mismatch(result)
        return result["answer"]

    def _checked_rag_stream(
        self,
        provider: str,
        model: str,
        query: str,
        system_prompt: str,
        filters,
    ) -> Generator:
        """Start a RAG stream with eager CONTENT_MISMATCH detection.

        Prefetches enough of the stream to check for CONTENT_MISMATCH before
        any chunks are yielded, so the caller can still raise an HTTP 400
        before the SSE response begins.  Returns a generator that replays the
        prefetched chunks then continues with the rest of the stream.
        """
        stream = self.llm_executor.rag_stream(
            provider=provider,
            model=model,
            query=query,
            system_prompt=system_prompt,
            filters=filters,
        )

        # Eagerly consume enough to detect CONTENT_MISMATCH
        buffer = ""
        prefetch: list = []
        for item in stream:
            if isinstance(item, TokenUsage):
                self.last_token_usage = item
                prefetch.append(item)
                break
            buffer += item
            prefetch.append(item)
            if len(buffer) >= 100:
                break

        if buffer.startswith("CONTENT_MISMATCH:"):
            raise ContentMismatchError(
                buffer[len("CONTENT_MISMATCH:") :].strip()
            )

        def _gen():
            yield from prefetch
            for item in stream:
                if isinstance(item, TokenUsage):
                    self.last_token_usage = item
                yield item

        return _gen()

    def make_outline_rag_stream(
        self, request: OutlineGenerateRequest
    ) -> Generator:
        sys_msg = self._system("outline.system.rag", None)
        usr_msg = self._system("outline.user", request.to_dict())

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

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
        sys_msg = self._system("presentation.system.rag", None)
        usr_msg = self._system("presentation.user", request.to_dict())

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        return self._checked_rag_stream(
            request.provider,
            request.model,
            usr_msg,
            sys_msg,
            filters if filters else None,
        )

    def generate_mindmap_with_rag(self, request: MindmapGenerateRequest):
        sys_msg = self._system(
            "mindmap.system.rag",
            request.to_dict(),
        )

        usr_msg = self._system(
            "mindmap.user",
            request.to_dict(),
        )

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        print(f"[DEBUG] RAG filters being applied: {filters}")
        print(
            f"[DEBUG] Request - subject: {request.subject}, grade: {request.grade} (type: {type(request.grade).__name__})"
        )

        result, token_usage = self.llm_executor.rag_batch(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            return_source_documents=True,
            filters=filters if filters else None,
            custom_prompt=None,
            verbose=False,
        )

        self.last_token_usage = token_usage
        self._check_content_mismatch(result)
        return result["answer"]

    def generate_mindmap_rag_stream(
        self, request: MindmapGenerateRequest
    ) -> Generator:
        sys_msg = self._system("mindmap.system.rag", request.to_dict())
        usr_msg = self._system("mindmap.user", request.to_dict())

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        return self._checked_rag_stream(
            request.provider,
            request.model,
            usr_msg,
            sys_msg,
            filters if filters else None,
        )
