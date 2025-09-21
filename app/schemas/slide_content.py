import uuid

from pydantic import BaseModel, Field, field_validator


# Request and Response models for outline generation
class OutlineGenerateRequest(BaseModel):
    topic: str = Field(..., description="The topic for the presentation")
    model: str = Field(..., description="The model to use for generation")
    provider: str = Field(..., description="The provider of the model")
    language: str = Field(..., description="The language for the presentation")
    slide_count: int = Field(
        ..., description="The number of slides to generate"
    )

    def to_dict(self):
        return {
            "topic": self.topic,
            "model": self.model,
            "provider": self.provider,
            "language": self.language,
            "slide_count": self.slide_count,
        }


# Request and Response models for presentation generation
class PresentationGenerateRequest(BaseModel):
    model: str = Field(..., description="The model to use for generation")
    provider: str = Field(..., description="The provider of the model")
    language: str = Field(..., description="The language for the presentation")
    slide_count: int = Field(
        ..., description="The number of slides to generate"
    )
    outline: str = Field(..., description="The outline for the presentation")

    def to_dict(self):
        return {
            "model": self.model,
            "provider": self.provider,
            "language": self.language,
            "slide_count": self.slide_count,
            "outline": self.outline,
        }
