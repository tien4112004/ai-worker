from typing import Any, Dict

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.repositories.llm_result_repository import llm_result_repository
from app.schemas.slide_content import OutlineGenerateRequest


class ContentService:
    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()

    def _system(self, key: str, vars: Dict[str, Any]) -> str:
        return self.prompt_store.render(key, vars)

    def make_slide_stream(self, topic: str):
        """Generate slide content using LLM and save result."""
        system_prompt = self._system(
            "slide.system",
            {
                "topic": topic,
            },
        )

        result = self.llm_executor.stream(
            provider="gemini",
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Create slide content for the topic: {topic}.",
                },
            ],
        )

        return result

    def make_outline_stream(self, request: OutlineGenerateRequest):
        system_prompt = self._system(
            "outline.system",
            {
                "topic": request.topic,
                "language": request.language,
                "slide_count": request.slide_count,
            },
        )

        result = self.llm_executor.stream(
            provider="gemini",
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Generate an outline for a presentation on the topic: {request.topic}. The presentation should be in {request.language} and consist of {request.slide_count} slides.",
                },
            ],
        )

        return result

    def make_presentation(self, topic: str):
        sys_msg = self._system(
            "outline.system",
            {
                "topic": topic,
                "language": "English",
                "slide_count": 5,
            },
        )

        outline = """1. Introduction to the topic
2. Key concepts and definitions
3. In-depth analysis and discussion
4. Case studies or examples
5. Conclusion and future directions
"""

        result = self.llm_executor.batch(
            provider="openai",
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": sys_msg},
                {
                    "role": "user",
                    "content": f"Create slide content for the topic: {topic} based on the following outline:\n{outline}",
                },
            ],
        )

        return result

    def make_outline(self, request: OutlineGenerateRequest):
        system_prompt = self._system(
            "outline.system",
            {
                "topic": request.topic,
                "language": request.language,
                "slide_count": request.slide_count,
            },
        )

        result = self.llm_executor.batch(
            provider="gemini",
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Generate an outline for a presentation on the topic: {request.topic}. The presentation should be in {request.language} and consist of {request.slide_count} slides.",
                },
            ],
        )

        return result
