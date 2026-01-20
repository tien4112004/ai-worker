from typing import Any

from fastapi import APIRouter, Depends

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.modification import (
    AIModificationResponse,
    ExpandSlideRequest,
    GenerateImageRequest,
    RefineContentRequest,
    RefineElementTextRequest,
    ReplaceElementImageRequest,
    SuggestThemeRequest,
    TransformLayoutRequest,
)
from app.services.modification_service import ModificationService

router = APIRouter(prefix="/modification", tags=["modification"])


def get_service():
    # In a real app, use dependency injection properly
    return ModificationService(LLMExecutor(), PromptStore())


@router.post("/refine", response_model=AIModificationResponse)
async def refine_content(
    request: RefineContentRequest,
    service: ModificationService = Depends(get_service),
):
    result = service.refine_content(request)
    return AIModificationResponse(success=True, data=result)


@router.post("/layout", response_model=AIModificationResponse)
async def transform_layout(
    request: TransformLayoutRequest,
    service: ModificationService = Depends(get_service),
):
    result = service.transform_layout(request)
    return AIModificationResponse(success=True, data=result)


@router.post("/image", response_model=AIModificationResponse)
async def generate_image(
    request: GenerateImageRequest,
    service: ModificationService = Depends(get_service),
):
    result = service.generate_image(request)
    # result is the image URL/Base64
    return AIModificationResponse(success=True, data={"url": result})


@router.post("/expand", response_model=AIModificationResponse)
async def expand_slide(
    request: ExpandSlideRequest,
    service: ModificationService = Depends(get_service),
):
    result = service.expand_slide(request)
    return AIModificationResponse(success=True, data=result)


@router.post("/refine-text", response_model=AIModificationResponse)
async def refine_element_text(
    request: RefineElementTextRequest,
    service: ModificationService = Depends(get_service),
):
    result = service.refine_element_text(request)
    return AIModificationResponse(success=True, data=result)


@router.post("/replace-image", response_model=AIModificationResponse)
async def replace_element_image(
    request: ReplaceElementImageRequest,
    service: ModificationService = Depends(get_service),
):
    result = service.replace_element_image(request)
    return AIModificationResponse(success=True, data=result)
