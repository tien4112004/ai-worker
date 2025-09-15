from typing import Optional

from pydantic import BaseModel, Field


class ImageGenerateRequest(BaseModel):
    prompt: str
    model: str
    provider: str
    sample_count: int = Field(
        default=1, description="Number of images to generate"
    )
    aspect_ratio: str = Field(
        default="1024x1024",
        description="Desired image dimensions, format: WIDTHxHEIGHT",
    )
    safety_filter_level: str = Field(
        default="BLOCK_MEDIUM",
        description="Safety filter level: BLOCK_NONE, BLOCK_LOW, BLOCK_MEDIUM, BLOCK_HIGH",
    )
    person_generation: str = Field(
        default="ALLOW",
        description="Allow or block person generation: ALLOW, BLOCK",
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducible generation"
    )

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "model": self.model,
            "provider": self.provider,
            "sample_count": self.sample_count,
            "aspect_ratio": self.aspect_ratio,
            "safety_filter_level": self.safety_filter_level,
            "person_generation": self.person_generation,
        }


class ImageGenerateResponse(BaseModel):
    base64_image: str = Field(
        description="Base64 encoded image data (without mime prefix)"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if image generation failed"
    )

    def to_dict(self):
        return {
            "base64_image": self.base64_image,
            "error": self.error,
        }
