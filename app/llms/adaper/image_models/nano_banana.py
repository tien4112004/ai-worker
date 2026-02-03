import base64
from datetime import datetime
from typing import Any, Dict

from google import genai
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

from app.core.config import settings
from app.llms.adaper.tracing import trace_span

logger = settings.logger


class NanoBananaAdapter:
    def __init__(self, model: str, api_key: str):
        """
        Initialize Nano Banana (Gemini 2.5 Flash Image) adapter.

        Args:
            model (str): Model name (e.g., 'gemini-2.5-flash-image-preview')
            api_key (str): Google Gemini API key
        """
        self.model = model
        self.client = genai.Client(api_key=api_key)

    @trace_span("nano_banana_generate", system="google")
    def generate(self, message: str, **params) -> Dict[str, Any]:
        """
        Generate image based on the provided prompt using Nano Banana.

        Args:
            message (str): The prompt for image generation.
            params: Additional parameters including:
                number_of_images (int): Number of images to generate.
                aspect_ratio (str): Desired image dimensions, format: 1:1, 9:16, 16:9, 4:3, 3:4
                safety_filter_level (str): Safety filter level.
                person_generation (Optional[Literal]): Allow or block person generation.
                seed (Optional[int]): Random seed for reproducible generation.
                negative_prompt (Optional[str]): Negative prompt to avoid certain elements.

        Returns:
            Dict[str, Any]: A dictionary containing either the base64 image data or an error message
        """
        try:
            number_of_images = params.get("number_of_images", 1)

            generation_config = {}
            if "seed" in params and params["seed"] is not None:
                generation_config["seed"] = params["seed"]

            call_kwargs = {
                "model": self.model,
                "contents": message,
            }
            if generation_config:
                call_kwargs["generation_config"] = generation_config

            response = self.client.models.generate_content(**call_kwargs)

            if not response or not response.candidates:
                return {
                    "error": "No image was generated in the response",
                    "base64_image": self._get_placeholder_image(),
                    "created": datetime.now().isoformat(),
                }

            images = []
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        try:
                            if (
                                hasattr(part, "inline_data")
                                and part.inline_data
                            ):
                                base64_data = base64.b64encode(
                                    part.inline_data.data
                                ).decode("utf-8")
                                images.append(base64_data)
                            elif hasattr(part, "text"):
                                continue
                        except Exception as e:
                            logger.warning(
                                f"Failed to process image part: {e}"
                            )
                            images.append(self._get_placeholder_image())

            if len(images) < number_of_images:
                for _ in range(number_of_images - len(images)):
                    images.append(
                        images[0] if images else self._get_placeholder_image()
                    )

            if not images:
                return {
                    "error": "No images found in response",
                    "base64_image": self._get_placeholder_image(),
                    "created": datetime.now().isoformat(),
                }

            span = trace.get_current_span()
            if response.usage_metadata:
                usage = response.usage_metadata
                if usage.prompt_token_count is not None:
                    span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                        usage.prompt_token_count,
                    )
                if usage.candidates_token_count is not None:
                    span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                        usage.candidates_token_count,
                    )
                if usage.total_token_count is not None:
                    span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                        usage.total_token_count,
                    )

            result_images = images[:number_of_images]
            span.set_attribute(
                SpanAttributes.OUTPUT_VALUE,
                f"{len(result_images)} image(s) generated",
            )

            return {
                "images": result_images,
                "count": len(result_images),
                "created": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Error during image generation with Nano Banana: {e}"
            )
            return {
                "error": str(e),
                "base64_image": self._get_placeholder_image(),
                "created": datetime.now().isoformat(),
            }

    def _get_placeholder_image(self) -> str:
        """Return a placeholder image for testing or error scenarios."""
        # 1x1 transparent PNG in base64
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
