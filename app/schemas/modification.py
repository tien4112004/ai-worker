from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RefineContentRequest(BaseModel):
    content: Any
    instruction: str
    context: Optional[Dict[str, Any]] = None


class TransformLayoutRequest(BaseModel):
    currentSchema: Dict[str, Any]
    targetType: str


class GenerateImageRequest(BaseModel):
    description: str
    style: str
    slideId: Optional[str] = None


class ExpandSlideRequest(BaseModel):
    currentSlide: Dict[str, Any]
    count: int = 2


class SuggestThemeRequest(BaseModel):
    mood: str


class RefineElementTextRequest(BaseModel):
    slideId: str
    elementId: str
    currentText: str
    instruction: str


class ReplaceElementImageRequest(BaseModel):
    slideId: str
    elementId: str
    description: str
    style: str


# Generic Response Wrapper
class AIModificationResponse(BaseModel):
    success: bool
    data: Any
    message: Optional[str] = None
