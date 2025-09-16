import base64
import json
from datetime import datetime
from typing import Any, Dict, List

from git import Union
from langchain_core.messages import BaseMessage
from langchain_google_vertexai.vision_models import VertexAIImageGeneratorChat

from app.core.config import settings

logger = settings.logger


class ImagenAdapter:
    def __init__(self, model: str, **params):
        # REQUIRES: VERTEX_PROJECT_ID and VERTEX_LOCATION set in env
        project_id = settings.project_id

        self._llm = VertexAIImageGeneratorChat(
            model_name=model,
            project=project_id,
            **params,
        )

    def _format_messages(
        self, messages: List[BaseMessage]
    ) -> Union[str, List[Union[str, Dict]]]:
        """Format messages for image generation."""
        # For simplicity, we'll just use the last user message as the prompt
        for message in reversed(messages):
            if message.type == "human":
                return message.content

        # Fallback to the last message content if no user message is found
        return messages[-1].content

    def generate(
        self, messages: List[BaseMessage], **params
    ) -> Dict[str, Any]:
        """Generate image based on the provided prompt."""
        # prompt = self._format_messages(messages)

        # Parse aspect ratio
        # aspect_ratio = params.get("aspect_ratio", "1024x1024")
        # if "x" in aspect_ratio:
        #     self._image_config["width"], self._image_config["height"] = map(
        #         int, aspect_ratio.split("x")
        #     )
        # else:
        #     self._image_config["width"] = self._image_config["height"] = int(
        #         aspect_ratio
        #     )

        try:
            # Generate the image
            response = self._llm.invoke(input=messages)

            # Extract base64 image data
            if response and hasattr(response, "content"):
                image_data = self._get_image_base64(response)
                return {
                    "base64_image": image_data,
                    "created": datetime.now().isoformat(),
                }

            return {
                "error": "No image was generated in the response",
                "base64_image": self._get_placeholder_image(),
                "created": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "error": str(e),
                "base64_image": self._get_placeholder_image(),
                "created": datetime.now().isoformat(),
            }

    def _get_placeholder_image(self) -> str:
        """Return a placeholder image for testing or error scenarios."""
        # 1x1 transparent PNG in base64
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    def _get_image_base64(self, response: BaseMessage) -> None:
        """Extract the base64 image string from the BaseMessage response."""
        image_block = next(
            block
            for block in response.content
            if isinstance(block, dict) and block.get("image_url")
        )
        return image_block["image_url"].get("url").split(",")[-1]
