import json
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from openai import APIError as OpenAIAPIError
from openai import (
    AuthenticationError,
    RateLimitError,
)

from app.core.config import settings
from app.core.exceptions import (
    AIAuthenticationError,
    AIRateLimitError,
    AIServiceError,
)
from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.modification import (
    ExpandCombinedTextRequest,
    RefineContentRequest,
    RefineElementTextRequest,
    ReplaceElementImageRequest,
    TransformLayoutRequest,
)

logger = logging.getLogger("uvicorn.error")


class ModificationService:
    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()

    def _render(self, key: str, vars: Dict[str, Any] | None) -> str:
        return self.prompt_store.render(key, vars)

    def _build_image_prompt(
        self,
        description: str,
        style: str,
        theme_description: Optional[str] = None,
        art_description: Optional[str] = None,
        slide_context: Optional[str] = None,
    ) -> str:
        """
        Build a detailed image generation prompt for presentation slides.

        DEPRECATED: This method is only used by the deprecated /api/modification/replace-image endpoint.
        Spring Boot backend now handles all prompt building and calls /api/image/generate directly.

        This method constructs a structured prompt by combining user description,
        art style, theme colors, and presentation requirements.

        Args:
            description: User's description of desired image
            style: Art style (e.g., "photorealistic", "cartoon", "digital-art")
            theme_description: Theme color modifiers from theme.modifiers
            art_description: Art style modifiers from art style configuration
            slide_context: Additional context about slide layout (optional) - no longer used

        Returns:
            Complete prompt string ready for image generation API
        """
        # Validate required fields
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        if not style or not style.strip():
            raise ValueError("Style cannot be empty")

        # Build prompt with structured format
        prompt_parts = [description]

        # Add style directive
        if style:
            prompt_parts.append(f"Art style: {style}")

        # Add art style modifiers (detailed instructions for the chosen style)
        if art_description:
            prompt_parts.append(art_description)

        # Add theme color modifiers
        if theme_description:
            prompt_parts.append(
                f"Color palette and theme: {theme_description}"
            )

        # Add presentation-specific requirements
        prompt_parts.append(
            "Create a professional, high-quality image suitable for presentation slides. "
            "Ensure clean composition, good contrast, and visual clarity."
        )

        # Add aspect ratio hint
        prompt_parts.append("Image format: 16:9 widescreen aspect ratio.")

        final_prompt = " ".join(prompt_parts)

        # Log warning if prompt is unusually long (monitoring only)
        if len(final_prompt) > 500:
            logger.warning(
                "Image prompt is unusually long (%d chars). Consider simplifying descriptions.",
                len(final_prompt),
            )

        return final_prompt

    def _get_operation(
        self, instruction: str, operation: Optional[str] = None
    ) -> str:
        """Determine operation type from explicit parameter or instruction text."""
        if operation:
            return operation.lower()

        # Fallback: parse instruction text
        instruction_lower = instruction.lower()
        if "expand" in instruction_lower or "more detail" in instruction_lower:
            return "expand"
        elif "shorten" in instruction_lower or "concise" in instruction_lower:
            return "shorten"
        elif "grammar" in instruction_lower or "spelling" in instruction_lower:
            return "grammar"
        elif "formal" in instruction_lower:
            return "formal"

        return "expand"  # Default fallback

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response, stripping markdown fences if present."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
            # Remove closing fence
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        return json.loads(cleaned)

    def refine_content(self, request: RefineContentRequest) -> Dict[str, Any]:
        try:
            slide_type = ""
            if request.context and request.context.slideType:
                slide_type = request.context.slideType

            # Determine which prompt to use based on operation
            operation = self._get_operation(
                request.instruction, request.operation
            )
            prompt_key = f"modification.slide.{operation}"

            prompt = self._render(
                prompt_key,
                {
                    "context_json": json.dumps(
                        request.schema, ensure_ascii=False
                    ),
                    "instruction": request.instruction,
                    "slide_type": slide_type,
                },
            )

            text, usage = self.llm_executor.batch(
                provider=request.provider,
                model=request.model,
                messages=[HumanMessage(content=prompt)],
            )
            logger.info(
                "refine_content tokens: %s",
                usage.total_tokens if usage else "N/A",
            )

            parsed = self._extract_json(text)
            return {"schema": parsed}
        except AuthenticationError as e:
            raise AIAuthenticationError(
                f"OpenAI authentication failed: {str(e)}"
            )
        except RateLimitError as e:
            raise AIRateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
        except OpenAIAPIError as e:
            raise AIServiceError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in refine_content")
            raise AIServiceError(f"Failed to refine content: {str(e)}")

    def transform_layout(
        self, request: TransformLayoutRequest
    ) -> Dict[str, Any]:
        try:
            prompt = self._render(
                "modification.slide.layout",
                {
                    "source_json": json.dumps(
                        request.currentSchema, ensure_ascii=False
                    ),
                    "target_type": request.targetType,
                },
            )

            text, usage = self.llm_executor.batch(
                provider=request.provider,
                model=request.model,
                messages=[HumanMessage(content=prompt)],
            )
            logger.info(
                "transform_layout tokens: %s",
                usage.total_tokens if usage else "N/A",
            )

            parsed = self._extract_json(text)
            return {"schema": parsed}
        except AuthenticationError as e:
            raise AIAuthenticationError(
                f"AI service authentication failed: {str(e)}"
            )
        except RateLimitError as e:
            raise AIRateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
        except OpenAIAPIError as e:
            raise AIServiceError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in transform_layout")
            raise AIServiceError(f"Failed to transform layout: {str(e)}")

    def refine_element_text(
        self, request: RefineElementTextRequest
    ) -> Dict[str, Any]:
        """Refine text content of a specific element."""
        try:
            slide_context = ""
            if request.slideType:
                slide_context += f"Slide layout type: {request.slideType}. "
            if request.slideSchema:
                title = request.slideSchema.get("title", "")
                if title:
                    slide_context += f"Slide title: {title}. "

            # Determine which prompt to use based on operation
            operation = self._get_operation(
                request.instruction, request.operation
            )
            prompt_key = f"modification.element.{operation}"

            prompt = self._render(
                prompt_key,
                {
                    "current_text": request.currentText,
                    "slide_context": slide_context,
                },
            )

            text, usage = self.llm_executor.batch(
                provider=request.provider,
                model=request.model,
                messages=[HumanMessage(content=prompt)],
            )
            logger.info(
                "refine_element_text tokens: %s",
                usage.total_tokens if usage else "N/A",
            )

            return {"refinedText": text.strip()}
        except AuthenticationError as e:
            raise AIAuthenticationError(
                f"AI service authentication failed: {str(e)}"
            )
        except RateLimitError as e:
            raise AIRateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
        except OpenAIAPIError as e:
            raise AIServiceError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in refine_element_text")
            raise AIServiceError(f"Failed to refine text: {str(e)}")

    def replace_element_image(
        self, request: ReplaceElementImageRequest
    ) -> Dict[str, Any]:
        """
        Replace image of a specific element.

        DEPRECATED: This endpoint is no longer called by Spring Boot backend.
        Spring Boot now builds the complete prompt and calls /api/image/generate directly.

        For backward compatibility, this method still works by building a prompt
        and calling the generic image generation endpoint.
        """
        try:
            logger.warning(
                "/api/modification/replace-image is deprecated. "
                "Spring Boot should call /api/image/generate directly with a built prompt."
            )

            # Build the image generation prompt
            image_prompt = self._build_image_prompt(
                description=request.description,
                style=request.style,
                theme_description=request.themeDescription,
                art_description=request.artDescription,
            )

            logger.info(
                "Built image prompt (%d chars) for deprecated endpoint",
                len(image_prompt),
            )

            # Delegate to generic image generation (which is the lightweight endpoint)
            image_result = self.llm_executor.generate_image(
                provider="google",
                model="gemini-2.5-flash-image",
                message=image_prompt,
                number_of_images=1,
                aspect_ratio="16:9",
            )

            # Extract base64 image and convert to data URI
            if (
                image_result
                and "images" in image_result
                and len(image_result["images"]) > 0
            ):
                base64_data = image_result["images"][0]
                image_url = f"data:image/png;base64,{base64_data}"
                return {"imageUrl": image_url}

            error_msg = image_result.get("error", "Image generation failed")
            raise RuntimeError(f"Image generation failed: {error_msg}")

        except ValueError as e:
            logger.error("Invalid input for image generation: %s", str(e))
            raise AIServiceError(f"Invalid input: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in replace_element_image")
            raise AIServiceError(f"Failed to replace image: {str(e)}")

    def expand_combined_text(
        self, request: ExpandCombinedTextRequest
    ) -> Dict[str, Any]:
        """Expand content of combined text items."""
        try:
            slide_context = ""
            if request.slideType:
                slide_context += f"Slide layout type: {request.slideType}. "
            if request.slideSchema:
                title = request.slideSchema.get("title", "")
                if title:
                    slide_context += f"Slide title: {title}. "

            # Convert items to JSON for the prompt
            items_json = json.dumps(request.items, ensure_ascii=False)

            # Determine which prompt to use based on operation
            operation = self._get_operation(
                request.instruction, request.operation
            )
            prompt_key = f"modification.combined_text.{operation}_items"

            prompt = self._render(
                prompt_key,
                {
                    "items_json": items_json,
                    "slide_context": slide_context,
                },
            )

            text, usage = self.llm_executor.batch(
                provider=request.provider,
                model=request.model,
                messages=[HumanMessage(content=prompt)],
            )
            logger.info(
                "refine_combined_text tokens: %s",
                usage.total_tokens if usage else "N/A",
            )

            # Parse the refined items from the response
            parsed = self._extract_json(text)
            return {"expandedItems": parsed}
        except AuthenticationError as e:
            raise AIAuthenticationError(
                f"AI service authentication failed: {str(e)}"
            )
        except RateLimitError as e:
            raise AIRateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
        except OpenAIAPIError as e:
            raise AIServiceError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in refine_combined_text")
            raise AIServiceError(f"Failed to refine combined text: {str(e)}")
