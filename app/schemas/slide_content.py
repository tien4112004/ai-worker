import uuid

from pydantic import BaseModel, Field, field_validator


# Request and Response models for outline generation
class OutlineGenerateRequest(BaseModel):
    topic: str
    model: str
    provider: str
    language: str
    slide_count: int
    learning_objective: str
    target_age: str

    def to_dict(self):
        return {
            "topic": self.topic,
            "model": self.model,
            "provider": self.provider,
            "language": self.language,
            "slide_count": self.slide_count,
            "learning_objective": self.learning_objective,
            "target_age": self.target_age,
        }


# Request and Response models for presentation generation
class PresentationGenerateRequest(BaseModel):
    model: str
    provider: str
    language: str
    slide_count: int
    learning_objective: str
    target_age: str
    outline: str

    def to_dict(self):
        return {
            "model": self.model,
            "provider": self.provider,
            "language": self.language,
            "slide_count": self.slide_count,
            "learning_objective": self.learning_objective,
            "target_age": self.target_age,
            "outline": self.outline,
        }
