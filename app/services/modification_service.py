import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.modification import (
    ExpandSlideRequest,
    GenerateImageRequest,
    RefineContentRequest,
    RefineElementTextRequest,
    ReplaceElementImageRequest,
    SuggestThemeRequest,
    TransformLayoutRequest,
)


class ModificationService:
    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()

    def _render(self, key: str, vars: Dict[str, Any] | None) -> str:
        return self.prompt_store.render(key, vars)

    def refine_content(self, request: RefineContentRequest) -> Any:
        prompt = self._render(
            "modification.refine",
            {
                "context_json": json.dumps(
                    request.content, ensure_ascii=False
                ),
                "instruction": request.instruction,
            },
        )

        # Using default model from settings or hardcoded high-quality model
        result = self.llm_executor.batch(
            provider="openai",
            model="gpt-4o",
            messages=[HumanMessage(content=prompt)],
        )
        # Result is expected to be a JSON string from the prompt contract
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Fallback or return raw string if parsing fails, but FE expects JSON
            return {"content": result}

    def transform_layout(self, request: TransformLayoutRequest) -> Any:
        prompt = self._render(
            "modification.layout",
            {
                "source_json": json.dumps(
                    request.currentSchema, ensure_ascii=False
                ),
                "target_type": request.targetType,
            },
        )

        result = self.llm_executor.batch(
            provider="openai",
            model="gpt-4o",
            messages=[HumanMessage(content=prompt)],
        )
        return json.loads(result)

    def generate_image(self, request: GenerateImageRequest) -> str:
        # First, generate the refined prompt
        prompt_generation_prompt = self._render(
            "modification.image",
            {
                "description": request.description,
                "context": request.slideId or "Presentation Slide",
                "style": request.style,
            },
        )

        optimized_prompt = self.llm_executor.batch(
            provider="openai",
            model="gpt-4o",
            messages=[HumanMessage(content=prompt_generation_prompt)],
        )

        # Then, generate the actual image
        # Assuming llm_executor has generate_image or we use DALL-E directly
        # The ContentService uses llm_executor.generate_image, let's use that

        # We need to Construct a message for generate_image as it expects it
        # But wait, ContentService.generate_image takes an ImageGenerateRequest
        # I'll just call the executor directly.

        image_result = self.llm_executor.generate_image(
            provider="openai",
            model="dall-e-3",
            message=optimized_prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            safety_filter_level="block_medium_and_above",
            person_generation="allow_adult",
            seed=None,
            negative_prompt="",
        )

        # image_result is likely a dict with "images" list
        if (
            image_result
            and "images" in image_result
            and len(image_result["images"]) > 0
        ):
            return image_result["images"][
                0
            ]  # Return the first image (url or base64)
        return ""

    def expand_slide(self, request: ExpandSlideRequest) -> Any:
        prompt = self._render(
            "modification.expand",
            {
                "slide_json": json.dumps(
                    request.currentSlide, ensure_ascii=False
                ),
                "count": request.count,
            },
        )

        result = self.llm_executor.batch(
            provider="openai",
            model="gpt-4o",
            messages=[HumanMessage(content=prompt)],
        )
        return json.loads(result)

    def refine_element_text(self, request: RefineElementTextRequest) -> Any:
        """Refine text content of a specific element"""
        prompt = self._render(
            "modification.refine_text",
            {
                "current_text": request.currentText,
                "instruction": request.instruction,
            },
        )

        result = self.llm_executor.batch(
            provider="openai",
            model="gpt-4o",
            messages=[HumanMessage(content=prompt)],
        )

        # Return plain text wrapped in expected format
        return {"refinedText": result.strip()}

    def replace_element_image(
        self, request: ReplaceElementImageRequest
    ) -> Any:
        """Replace image of a specific element"""
        # First, generate the optimized DALL-E prompt
        prompt = self._render(
            "modification.replace_image",
            {"description": request.description, "style": request.style},
        )

        optimized_prompt = self.llm_executor.batch(
            provider="openai",
            model="gpt-4o",
            messages=[HumanMessage(content=prompt)],
        )

        # Then generate the actual image
        image_result = self.llm_executor.generate_image(
            provider="openai",
            model="dall-e-3",
            message=optimized_prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            safety_filter_level="block_medium_and_above",
            person_generation="allow_adult",
            seed=None,
            negative_prompt="",
        )

        # Extract image URL
        if (
            image_result
            and "images" in image_result
            and len(image_result["images"]) > 0
        ):
            return {"imageUrl": image_result["images"][0]}
        return {"imageUrl": ""}
