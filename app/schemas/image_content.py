from pydantic import BaseModel

class ImageGenerateRequest(BaseModel):
    prompt: str
    sample_count: int
    aspect_ratio: str
    safety_filter_level: str
    person_generation: str
    seed: int

class ImageGenerateResponse(BaseModel):
    image_uri: str
    mime_type: str
    prompt: str
    created: str