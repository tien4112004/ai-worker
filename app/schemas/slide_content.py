import uuid

from pydantic import BaseModel, Field, field_validator


# Request and Response models for outline generation
class OutlineGenerateRequest(BaseModel):
    topic: str
    model: str
    provider: str
    language: str
    slide_count: int

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
    model: str
    provider: str
    language: str
    slide_count: int
    outline: str

    def to_dict(self):
        return {
            "model": self.model,
            "provider": self.provider,
            "language": self.language,
            "slide_count": self.slide_count,
            "outline": self.outline,
        }
