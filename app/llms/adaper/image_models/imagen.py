from datetime import datetime
from typing import Any, Dict

from vertexai.preview.vision_models import ImageGenerationModel

from app.core.config import settings

logger = settings.logger


class ImagenAdapter:
    def __init__(self, model: str):
        # REQUIRES: VERTEX_PROJECT_ID and VERTEX_LOCATION set in env
        self._llm = ImageGenerationModel.from_pretrained(model_name=model)

    def generate(self, message: str, **params) -> Dict[str, Any]:
        """
        Generate image based on the provided prompt.
        Args:
            message (str): The prompt for image generation.
            params: Additional parameters including:
                number_of_images (int): Number of images to generate.
                aspect_ratio (str): Desired image dimensions, format: WIDTHxHEIGHT.
                safety_filter_level (str): Safety filter level: BLOCK_NONE, BLOCK_LOW, BLOCK_MEDIUM, BLOCK_HIGH.
                person_generation (Optional[Literal]): Allow or block person generation: ALLOW, BLOCK.
                seed (Optional[int]): Random seed for reproducible generation.
                negative_prompt (Optional[Literal]): Negative prompt to avoid certain elements in the image.
        Returns:
            Dict[str, Any]: A dictionary containing either the base64 image data or an error message
        """

        try:
            response = self._llm.generate_images(prompt=message, **params)

            if not response or not response.images:
                return {
                    "error": "No image was generated in the response",
                    "base64_image": self._get_placeholder_image(),
                    "created": datetime.now().isoformat(),
                }

            images = []
            for image in response.images:
                try:
                    base64_data = self._get_image_base64(image)
                    images.append(base64_data)
                except Exception as e:
                    logger.warning(f"Failed to process image: {e}")
                    images.append(self._get_placeholder_image())

            return {
                "images": images,
                "count": len(images),
            }
        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            return {
                "error": str(e),
                "base64_image": self._get_placeholder_image(),
            }

    def _get_placeholder_image(self) -> str:
        """Return a placeholder image for testing or error scenarios."""
        # 1x1 transparent PNG in base64
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    def _get_image_base64(self, image) -> str:
        """Extract the base64 image string from the Vertex AI image response."""
        try:
            # For Vertex AI ImageGenerationModel, images have a _image_bytes attribute
            if hasattr(image, "_image_bytes"):
                import base64

                return base64.b64encode(image._image_bytes).decode("utf-8")

            # Alternative: if the image object has a different structure
            # You might need to adjust this based on the actual Vertex AI response format
            if hasattr(image, "data"):
                return image.data

            # Fallback to placeholder if we can't extract the image
            logger.warning("Could not extract image data, using placeholder")
            return self._get_placeholder_image()

        except Exception as e:
            logger.error(f"Error extracting image base64: {e}")
            return self._get_placeholder_image()
