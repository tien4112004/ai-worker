from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RefineContext(BaseModel):
    slideId: Optional[str] = None
    slideType: Optional[str] = None


class RefineContentRequest(BaseModel):
    schema: Dict[str, Any]
    instruction: str
    context: Optional[RefineContext] = None
    operation: Optional[str] = None  # "expand", "shorten", "grammar", "formal"
    model: str = Field(..., description="The model to use for modification")
    provider: str = Field(..., description="The provider of the model")


class TransformLayoutRequest(BaseModel):
    currentSchema: Dict[str, Any]
    targetType: str
    model: str = Field(..., description="The model to use for modification")
    provider: str = Field(..., description="The provider of the model")


class RefineElementTextRequest(BaseModel):
    slideId: str
    elementId: str
    currentText: str
    instruction: str
    slideSchema: Optional[Dict[str, Any]] = None
    slideType: Optional[str] = None
    operation: Optional[str] = None  # "expand", "shorten", "grammar", "formal"
    model: str = Field(..., description="The model to use for modification")
    provider: str = Field(..., description="The provider of the model")


class ReplaceElementImageRequest(BaseModel):
    slideId: str
    elementId: str
    description: str
    style: str
    themeDescription: Optional[str] = None
    artDescription: Optional[str] = None
    slideSchema: Optional[Dict[str, Any]] = None
    slideType: Optional[str] = None
    model: str = Field(..., description="The model to use for text generation")
    provider: str = Field(..., description="The provider for text generation")


class ExpandCombinedTextRequest(BaseModel):
    slideId: str
    items: List[Any]  # Can be strings or dicts
    instruction: str
    slideSchema: Optional[Dict[str, Any]] = None
    slideType: Optional[str] = None
    operation: Optional[str] = None  # "expand", "shorten", "grammar", "formal"
    model: str = Field(..., description="The model to use for expansion")
    provider: str = Field(..., description="The provider of the model")


# Generic Response Wrapper
class AIModificationResponse(BaseModel):
    success: bool
    data: Any
    message: Optional[str] = None
