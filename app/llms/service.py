# app/llms/service.py
import os
from typing import Generator, Optional, Dict, Any, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import Cast

from app.llms.factory import LLMFactory
from app.core.config import settings
from langchain_core.messages import AIMessageChunk


class LLMService:
    def __init__(self, llm_factory: LLMFactory, model_name: Optional[str] = None):
        self.model_name = model_name or settings.default_model
        self._setup_environment()
        self.factory = llm_factory
        self._llm = None

    def _setup_environment(self):
        """Set up environment variables for API keys."""
        if settings.google_api_key:
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

    @property
    def llm(self) -> BaseChatModel:
        """Lazy loading of the LLM instance."""
        if self._llm is None:
            self._llm = self.factory.build_from_model_string(self.model_name)
        return self._llm

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None):
        """Generate text using the configured LLM."""
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        try:
            response = self.llm.stream(messages)
            # content = response.content
            # # Handle both string and list content types
            # if isinstance(content, list):
            #     return " ".join(str(item) for item in content)
            return response
            # return """
            # Lorem, ipsum dolor sit amet consectetur adipisicing elit. Ducimus omnis accusantium molestias ipsum modi eveniet, 
            # qui vel quaerat labore at dicta numquam quos laudantium, amet perspiciatis. Vero eveniet veritatis aliquam.
            # """
        except Exception as e:
            return f"Error generating content: {str(e)}"

    def generate_slide_content(self, topic: str, slide_count: int = 5, learning_objective: str = "", target_age: str = ""):
        """Generate slide content for a given topic."""
        system_prompt = f"""You are an expert content creator specializing in educational presentations.
        Create engaging and informative slide content that is appropriate for the target audience.
        
        Guidelines:
        - Create exactly {slide_count} slides
        - Make content appropriate for {target_age if target_age else 'general audience'}
        - Focus on the learning objective: {learning_objective if learning_objective else 'comprehensive understanding'}
        - Use clear, concise language
        - Include engaging examples where appropriate
        """
        
        user_prompt = f"""Create a {slide_count}-slide presentation about: {topic}
        
        Learning objective: {learning_objective if learning_objective else 'Provide comprehensive understanding'}
        Target audience: {target_age if target_age else 'General audience'}
        
        Format the output as follows:
        Slide 1: [Title]
        - Content point 1
        - Content point 2
        
        Slide 2: [Title]
        - Content point 1
        - Content point 2
        
        Continue for all {slide_count} slides."""
        
        return self.generate_text(user_prompt, system_prompt)

    def generate_outline(self, topic: str):
        """Generate an outline for a given topic."""
        system_prompt = """You are an expert content organizer. Create well-structured outlines that break down complex topics into logical, easy-to-follow sections."""
        
        user_prompt = f"""Create a detailed outline for the topic: {topic}
        
        The outline should include:
        1. Main sections with clear headings
        2. Sub-sections with key points
        3. Logical flow from introduction to conclusion
        4. Suggestions for examples or case studies where appropriate
        
        Format the outline clearly with appropriate indentation and numbering."""
    
        # return self.generate_text(user_prompt, system_prompt)
            
        # Mock response as Iterator[BaseMessageChunk]
        mock_content = f"""
            I. Introduction to {topic}
            A. Definition and scope
            B. Importance and relevance
            
            II. Main Concepts
            A. Key principles
            B. Core components
            
            III. Practical Applications
            A. Real-world examples
            B. Case studies
            
            IV. Conclusion
            A. Summary of key points
            B. Future considerations
            """
        # return iter([AIMessageChunk(content=mock_content)])

        return mock_content
