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
    targetAge: str


class OutlineGenerateResponse(BaseModel):
    result: str


# Request and Response models for presentation generation
class PresentationGenerateRequest(BaseModel):
    model: str
    provider: str
    language: str
    slide_count: int
    learning_objective: str
    targetAge: str
    outline: str
