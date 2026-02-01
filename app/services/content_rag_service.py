import base64
import os
import random
from asyncio import sleep
from typing import Any, Dict, Generator, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.image_content import ImageGenerateRequest
from app.schemas.mindmap_content import MindmapGenerateRequest
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)
from app.schemas.token_usage import TokenUsage


class ContentRagService:
    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()
        self.last_token_usage = None

    def _system(self, key: str, vars: Dict[str, Any] | None) -> str:
        return self.prompt_store.render(key, vars)

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
        return result

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
        return result

    def make_outline_rag_stream(
        self, request: OutlineGenerateRequest
    ) -> Tuple[List[str], TokenUsage]:
        result = self.make_outline_with_rag(request)
        return [result["answer"]], self.last_token_usage

    def make_presentation_rag_stream(
        self, request: PresentationGenerateRequest
    ) -> Tuple[List[str], TokenUsage]:
        result = self.make_presentation_with_rag(request)
        return [result["answer"]], self.last_token_usage

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
        return result

    def generate_mindmap_rag_stream(
        self, request: MindmapGenerateRequest
    ) -> Tuple[List[str], TokenUsage]:
        result = self.generate_mindmap_with_rag(request)
        return [result["answer"]], self.last_token_usage
