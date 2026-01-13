import base64
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.depends import ContentServiceDep
from app.schemas.image_content import (
    ImageGenerateRequest,
    ImageGenerateResponse,
)
from app.schemas.mindmap_content import MindmapGenerateRequest
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)
from app.schemas.token_usage import TokenUsage
from app.utils.server_sent_event import sse_json_by_json, sse_word_by_word


logger = logging.getLogger(__name__)


class GenerateResponse(BaseModel):
    """Generic response wrapper with token usage."""
    data: Any
    token_usage: TokenUsage | None = None


router = APIRouter(tags=["generate"])


@router.post("/outline/generate")
def generateOutline(
    outlineGenerateRequest: OutlineGenerateRequest, svc: ContentServiceDep
):
    result = svc.make_outline(outlineGenerateRequest)
    token_usage = svc.last_token_usage
    logger.info(f"[OUTLINE/GENERATE] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}")
    return GenerateResponse(data=result, token_usage=token_usage)


@router.post("/outline/generate/stream")
def generateOutline_Stream(
    request: Request,
    outlineGenerateRequest: OutlineGenerateRequest,
    svc: ContentServiceDep,
):
    chunks, token_usage = svc.make_outline_stream(outlineGenerateRequest)
    logger.info(f"[OUTLINE/GENERATE/STREAM] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}")
    print("Starting outline stream response")
    return EventSourceResponse(sse_word_by_word(request, chunks, token_usage), ping=None)


@router.post("/presentations/generate")
def generatePresentation(
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentServiceDep,
):
    result = svc.make_presentation(presentationGenerateRequest)
    token_usage = svc.last_token_usage
    logger.info(f"[PRESENTATIONS/GENERATE] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}")
    return GenerateResponse(data=result, token_usage=token_usage)


@router.post("/presentations/generate/stream")
def generatePresentation_Stream(
    request: Request,
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received presentation stream request:", presentationGenerateRequest)

    chunks, token_usage = svc.make_presentation_stream(presentationGenerateRequest)
    logger.info(f"[PRESENTATIONS/GENERATE/STREAM] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}")

    return EventSourceResponse(sse_json_by_json(request, chunks, token_usage), ping=None)


# Mock endpoints for testing without LLM calls
@router.post("/outline/generate/mock")
def generateOutline_Mock(
    svc: ContentServiceDep, outlineGenerateRequest: OutlineGenerateRequest
):
    print("Received mock outline request:", outlineGenerateRequest)
    result, token_usage = svc.make_outline_mock(outlineGenerateRequest)
    logger.info(f"[OUTLINE/GENERATE/MOCK] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}")
    return GenerateResponse(data=result, token_usage=token_usage)


@router.post("/outline/generate/stream/mock")
async def generateOutline_Mock_Stream(
    request: Request,
    outlineGenerateRequest: OutlineGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received mock outline stream request:", outlineGenerateRequest)
    chunks, token_usage = svc.make_outline_stream_mock()
    logger.info(f"[OUTLINE/GENERATE/STREAM/MOCK] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}")

    async def event_stream():
        for chunk in chunks:
            if await request.is_disconnected():
                break
            yield {
                "data": base64.b64encode(chunk.encode("utf-8")).decode("ascii")
            }

    return EventSourceResponse(sse_word_by_word(request, chunks, token_usage), media_type="text/event-stream")


@router.post("/presentations/generate/mock")
def generatePresentation_Mock(
    svc: ContentServiceDep,
    presentationGenerateRequest: PresentationGenerateRequest,
):
    print("Received mock presentation request:", presentationGenerateRequest)
    result, token_usage = svc.make_presentation_mock(presentationGenerateRequest)
    logger.info(f"[PRESENTATIONS/GENERATE/MOCK] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}")
    return GenerateResponse(data=result, token_usage=token_usage)


@router.post("/presentations/generate/stream/mock")
async def generatePresentation_Mock_Stream(
    request: Request,
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received mock presentation stream request:", presentationGenerateRequest)

    slides, token_usage = await svc.make_presentation_stream_mock()
    logger.info(f"[PRESENTATIONS/GENERATE/STREAM/MOCK] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}")
    
    # Convert slide dicts to JSON strings for sse_json_by_json
    slide_strings = [json.dumps(slide, ensure_ascii=False) for slide in slides]

    return EventSourceResponse(sse_json_by_json(request, slide_strings, token_usage), media_type="text/event-stream")


@router.post("/image/generate", response_model=ImageGenerateResponse)
def generate_image(
    imageGenerateRequest: ImageGenerateRequest, svc: ContentServiceDep
):
    print("Received image generation request:", imageGenerateRequest)

    result = svc.generate_image(imageGenerateRequest)
    if "error" in result and result["error"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    logger.info(f"[IMAGE/GENERATE] Images generated: count={result['count']}, model={imageGenerateRequest.model} (token_usage not available for image generation)")
    return {
        "images": result["images"],
        "count": result["count"],
        "error": None,
        "token_usage": None,
    }


@router.post("/image/generate/mock", response_model=ImageGenerateResponse)
def generate_image_mock(
    imageGenerateRequest: ImageGenerateRequest, svc: ContentServiceDep
):
    print("Received mock image generation request:", imageGenerateRequest)

    result = svc.generate_image_mock(imageGenerateRequest)
    if "error" in result and result["error"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    logger.info(f"[IMAGE/GENERATE/MOCK] Images generated: count={result['count']}, model={imageGenerateRequest.model} (token_usage not available for image generation)")
    return {
        "images": result["images"],
        "count": result["count"],
        "error": None,
        "token_usage": None,
    }


@router.post("/mindmap/generate")
def generateMindmap(
    mindmapGenerateRequest: MindmapGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received mindmap generation request:", mindmapGenerateRequest)
    result = svc.generate_mindmap(mindmapGenerateRequest)
    token_usage = svc.last_token_usage
    logger.info(f"[MINDMAP/GENERATE] Token Usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}, model={token_usage.model}")
    return GenerateResponse(data=result, token_usage=token_usage)


@router.post("/mindmap/generate/mock")
def generateMindmap_Mock(
    svc: ContentServiceDep,
    mindmapGenerateRequest: MindmapGenerateRequest,
):
    print("Received mock mindmap generation request:", mindmapGenerateRequest)
    result, token_usage = svc.generate_mindmap_mock(mindmapGenerateRequest)
    return GenerateResponse(data=result, token_usage=token_usage)
