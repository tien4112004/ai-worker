import uuid
from pydantic import BaseModel, Field, field_validator

class OutlineGenerateRequest(BaseModel):
    topic: str
    model: str
    language: str
    slide_count: int
    learning_objective: str
    targetAge: str

class OutlineGenerateResponse(BaseModel):
    result: str

class User(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    email: str

    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v