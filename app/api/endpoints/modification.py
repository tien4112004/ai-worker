import logging

from fastapi import APIRouter, Depends

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.modification import (
    AIModificationResponse,
    ExpandCombinedTextRequest,
    ExpandNodeRequest,
    RefineBranchRequest,
    RefineContentRequest,
    RefineElementTextRequest,
    RefineNodeRequest,
    ReplaceElementImageRequest,
    TransformLayoutRequest,
)
from app.services.modification_service import ModificationService

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/modification", tags=["modification"])


def get_service():
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


@router.post("/refine-text", response_model=AIModificationResponse)
async def refine_element_text(
    request: RefineElementTextRequest,
    service: ModificationService = Depends(get_service),
):
    result = service.refine_element_text(request)
    return AIModificationResponse(success=True, data=result)


@router.post(
    "/replace-image", response_model=AIModificationResponse, deprecated=True
)
async def replace_element_image(
    request: ReplaceElementImageRequest,
    service: ModificationService = Depends(get_service),
):
    """
    DEPRECATED: Replace image of a specific element.

    This endpoint is no longer called by Spring Boot. The backend now builds
    the complete prompt and calls /api/image/generate directly.

    This endpoint is kept for backward compatibility only.
    """
    result = service.replace_element_image(request)
    return AIModificationResponse(success=True, data=result)


@router.post("/refine-combined-text", response_model=AIModificationResponse)
async def refine_combined_text(
    request: ExpandCombinedTextRequest,
    service: ModificationService = Depends(get_service),
):
    result = service.expand_combined_text(request)
    return AIModificationResponse(success=True, data=result)


# Mindmap modification endpoints
@router.post("/mindmap/refine-node", response_model=AIModificationResponse)
async def refine_mindmap_node(
    request: RefineNodeRequest,
    service: ModificationService = Depends(get_service),
):
    """Refine a mindmap node's content (expand, shorten, fix grammar, formalize)."""
    result = service.refine_mindmap_node(request)
    return AIModificationResponse(success=True, data=result)


@router.post("/mindmap/expand-node", response_model=AIModificationResponse)
async def expand_mindmap_node(
    request: ExpandNodeRequest,
    service: ModificationService = Depends(get_service),
):
    """Generate child nodes for a mindmap node with AI."""
    result = service.expand_mindmap_node(request)
    return AIModificationResponse(success=True, data=result)


@router.post("/mindmap/refine-branch", response_model=AIModificationResponse)
async def refine_mindmap_branch(
    request: RefineBranchRequest,
    service: ModificationService = Depends(get_service),
):
    """Refine multiple nodes in a mindmap branch together."""
    result = service.refine_mindmap_branch(request)
    return AIModificationResponse(success=True, data=result)
