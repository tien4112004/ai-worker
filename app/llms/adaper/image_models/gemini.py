import base64
import os
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core import consts
from app.core.config import settings

logger = settings.logger


class GeminiImageAdapter:
    def __init__(self, model: str, **params):
        # REQUIRES: GOOGLE_API_KEY in env
        self.model_name = model
        self.params = params

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            logger.error("missing GOOGLE_API_KEY")
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        # Initialize the LangChain Google Generative AI model
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Format messages for image generation."""
        # For simplicity, we'll just use the last user message as the prompt
        for message in reversed(messages):
            if message.type == "human":
                return message.content

        # Fallback to the last message content if no user message is found
        return messages[-1].content

    def generate(
        self, model: str, messages: List[BaseMessage], **params
    ) -> Dict[str, Any]:
        """Generate image based on the provided prompt."""
        # prompt = self._format_messages(messages)

        # Parse aspect ratio
        aspect_ratio = params.get("aspect_ratio", "1024x1024")
        if "x" in aspect_ratio:
            width, height = map(int, aspect_ratio.split("x"))
        else:
            width = height = int(aspect_ratio)

        # # Add safety settings
        # safety_level = params.get("safety_filter_level", "BLOCK_MEDIUM")
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_level},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_level},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_level},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_level},
        # ]

        try:
            image_model = ChatGoogleGenerativeAI(
                model=model or self.model_name,
                google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
                convert_system_message_to_human=True,
                generation_config={
                    "temperature": settings.llm_temperature,
                    "top_p": 1.0,
                    "top_k": 32,
                    "max_output_tokens": 2048,
                    "response_mime_type": "image/png",
                },
                max_retries=settings.max_retries,
                # safety_settings=safety_settings,
                image_size={"width": width, "height": height},
            )

            # Generate the image
            response = image_model.invoke(
                input=messages,
                # generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
            )

            # Based on the response, handle the image data
            print(response)  # Debug print
            # Process the response which should contain image data
            if response and hasattr(response, "content"):
                # For Gemini, the image can be in different formats depending on the response
                if isinstance(response.content, list):
                    # Find the image block in the content list
                    for block in response.content:
                        if isinstance(block, dict) and "image_url" in block:
                            # Extract base64 data from the URL
                            image_data = block["image_url"]["url"].split(",")[
                                -1
                            ]
                            image_bytes = base64.b64decode(image_data)

                            # Save the image to a file
                            with open("response_image.png", "wb") as f:
                                f.write(image_bytes)

                            return {
                                "base64_image": image_data,
                                "created": datetime.now().isoformat(),
                            }

                # If response.content is a string containing a base64 image
                elif (
                    isinstance(response.content, str)
                    and "base64" in response.content
                ):
                    image_data = response.content.split(",")[-1]
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
            # Return error with placeholder image
            return {
                "error": str(e),
                "base64_image": self._get_placeholder_image(),
                "created": datetime.now().isoformat(),
            }

    def _get_placeholder_image(self) -> str:
        """Return a placeholder image for testing or error scenarios."""
        # 1x1 transparent PNG in base64
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    def _get_image_base64(response: AIMessage) -> None:
        """"""
        image_block = next(
            block
            for block in response.content
            if isinstance(block, dict) and block.get("image_url")
        )
        return image_block["image_url"].get("url").split(",")[-1]
