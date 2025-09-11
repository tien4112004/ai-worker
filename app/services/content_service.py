from typing import Any, Dict, Generator

from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.repositories.llm_result_repository import llm_result_repository
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)


class ContentService:
    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()

    def _system(self, key: str, vars: Dict[str, Any]) -> str:
        return self.prompt_store.render(key, vars)

    # Presentation Generation
    def make_presentation_stream(self, request: PresentationGenerateRequest):
        """Generate slide content using LLM and save result.
        Args:
            request (PresentationGenerateRequest): Request object containing parameters for slide generation.
        Returns:
            Generator: A generator yielding parts of the generated slide content.
        """
        system_prompt = self._system(
            "presentation.system",
            {},
        )

        user_prompt = self._system(
            "presentation.user",
            {
                "outline": request.outline,
                "language": request.language,
                "slide_count": request.slide_count,
                "learning_objective": request.learning_objective,
                "targetAge": request.targetAge,
            },
        )

        result = self.llm_executor.stream(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
        )

        return result

    def make_presentation(self, request: PresentationGenerateRequest):
        """
        Generate slide content using LLM and save result.
        Args:
            request (PresentationGenerateRequest): Request object containing parameters for slide generation.
        Returns:
            Dict: A dictionary containing the generated slide content.
        """
        sys_msg = self._system(
            "outline.system",
            {
                "topic": request.outline,
                "language": request.language,
                "slide_count": request.slide_count,
                "learning_objective": request.learning_objective,
                "targetAge": request.targetAge,
                "outline": request.outline,
            },
        )

        result = self.llm_executor.batch(
            provider=request.provider,
            model=request.model,
            messages=[{"role": "system", "content": sys_msg}],
        )

        return result

    # Outline Generation
    def make_outline_stream(self, request: OutlineGenerateRequest):
        """Generate outline using LLM and save result.
        Args:
            request (OutlineGenerateRequest): Request object containing parameters for outline generation.
        Returns:
            Generator: A generator yielding parts of the generated outline.
        """
        system_prompt = self._system("outline.system", {})

        user_prompt = self._system(
            "outline.user",
            {
                "topic": request.topic,
                "language": request.language,
                "slide_count": request.slide_count,
                "learning_objective": request.learning_objective,
                "targetAge": request.targetAge,
            },
        )

        result = self.llm_executor.stream(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
        )

        return result

    def make_outline(self, request: OutlineGenerateRequest):
        """Generate outline using LLM and save result.
        Args:
            request (OutlineGenerateRequest): Request object containing parameters for outline generation.
        Returns:
            Dict: A dictionary containing the generated outline.
        """
        system_prompt = self._system(
            "outline.system",
            {
                "topic": request.topic,
                "language": request.language,
                "slide_count": request.slide_count,
            },
        )

        result = self.llm_executor.batch(
            provider=request.provider,
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

    def make_presentation_mock(self):
        """Generate mock slide content for testing purposes.
        Returns:
            Dict: A dictionary containing the mock slide content.
        """
        result = """{
    "title": "Introduction to AI",
    "content": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
}

{
    "title": "History of AI",
    "content": "The concept of AI dates back to ancient times, but the field formally began in the 1950s with the development of early computers.",
}

{
    "title": "Types of AI",
    "content": "There are two main types of AI: Narrow AI, which is designed for specific tasks, and General AI, which has the ability to perform any intellectual task that a human can do.",
}"""
        return result

    def make_presentation_stream_mock(self):
        """Generate mock slide content stream for testing purposes.
        Yields:
            str: Parts of the mock slide content.
        """
        slides = [
            {
                "title": "Introduction to AI",
                "content": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
            },
            {
                "title": "History of AI",
                "content": "The concept of AI dates back to ancient times, but the field formally began in the 1950s with the development of early computers.",
            },
            {
                "title": "Types of AI",
                "content": "There are two main types of AI: Narrow AI, which is designed for specific tasks, and General AI, which has the ability to perform any intellectual task that a human can do.",
            },
        ]
        import json
        import time

        for slide in slides:
            # Yield properly formatted JSON that matches the expected slide structure
            json_str = json.dumps(
                {
                    "title": slide["title"],
                    "points": [
                        slide["content"]
                    ],  # Convert content to points array as expected by the system prompt
                }
            )
            yield json_str
            # Optional: Add small delay to simulate real streaming
            time.sleep(0.1)

    def make_outline_stream_mock(self) -> Generator[str, None, None]:
        """Generate mock outline stream for testing purposes.
        Yields:
            str: Parts of the mock outline.
        """
        import re
        import time

        outline = """```md
# Introduction to Artificial Intelligence
• What is AI?
• Why AI matters in today's world
• Overview of presentation goals

# History and Evolution of AI
• Early concepts and origins (1940s-1950s)
• The AI winters and revivals
• Major breakthroughs and milestones
• Key pioneers and their contributions

# Types and Categories of AI
• Narrow AI vs General AI
• Machine Learning fundamentals
• Deep Learning and Neural Networks
• Natural Language Processing
• Computer Vision

# Real-World Applications of AI
• Healthcare and medical diagnosis
• Transportation and autonomous vehicles
• Finance and fraud detection
• Entertainment and recommendation systems
• Smart homes and IoT devices

# Future of AI and Emerging Trends
• Ethical considerations and responsible AI
• AI governance and regulation
• Potential societal impacts
• Career opportunities in AI
• Preparing for an AI-driven future
```"""
        # Split the outline into meaningful chunks (words and punctuation)
        chunks = re.findall(r"\S+|\s+", outline)

        for chunk in chunks:
            yield chunk
            # Optional: Add small delay to simulate real streaming
            time.sleep(0.01)

    def make_outline_mock(self):
        """Generate mock outline for testing purposes.
        Returns:
            Dict: A dictionary containing the mock outline.
        """
        result = """```md
# Introduction to Artificial Intelligence
• What is AI?
• Why AI matters in today's world
• Overview of presentation goals

# History and Evolution of AI
• Early concepts and origins (1940s-1950s)
• The AI winters and revivals
• Major breakthroughs and milestones
• Key pioneers and their contributions

# Types and Categories of AI
• Narrow AI vs General AI
• Machine Learning fundamentals
• Deep Learning and Neural Networks
• Natural Language Processing
• Computer Vision

# Real-World Applications of AI
• Healthcare and medical diagnosis
• Transportation and autonomous vehicles
• Finance and fraud detection
• Entertainment and recommendation systems
• Smart homes and IoT devices

# Future of AI and Emerging Trends
• Ethical considerations and responsible AI
• AI governance and regulation
• Potential societal impacts
• Career opportunities in AI
• Preparing for an AI-driven future
```"""
        return result
