import logging
from typing import Any

from fastapi import APIRouter, logger
from pydantic import BaseModel

from app.depends import ContentServiceDep
from app.schemas.slide_content import OutlineGenerateRequest
from app.schemas.token_usage import TokenUsage

logger = logging.getLogger(__name__)


class GenerateResponse(BaseModel):
    """Generic response wrapper with token usage."""

    data: Any
    token_usage: TokenUsage | None = None


router = APIRouter(tags=["generate"])


@router.post("/outline/rag/generate")
def generate_outline_with_rag(
    outlineGenerateRequest: OutlineGenerateRequest, svc: ContentServiceDep
):
    result = svc.make_outline_with_rag(outlineGenerateRequest)
    token_usage = svc.last_token_usage
    logger.info(
        f"[OUTLINE/RAG/GENERATE] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}"
    )
    return GenerateResponse(data=result, token_usage=token_usage)
