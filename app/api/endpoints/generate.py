import base64
import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
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
from app.utils.server_sent_event import sse_json_by_json, sse_word_by_word

router = APIRouter(tags=["generate"])


@router.post("/outline/generate")
def generateOutline(
    outlineGenerateRequest: OutlineGenerateRequest, svc: ContentServiceDep
):
    result = svc.make_outline(outlineGenerateRequest)
    return result


@router.post("/outline/generate/stream")
def generateOutline_Stream(
    request: Request,
    outlineGenerateRequest: OutlineGenerateRequest,
    svc: ContentServiceDep,
):
    result = svc.make_outline_stream(outlineGenerateRequest)
    print("Starting outline stream response")
    return EventSourceResponse(sse_word_by_word(request, result), ping=None)


@router.post("/presentations/generate")
def generatePresentation(
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentServiceDep,
):
    result = svc.make_presentation(presentationGenerateRequest)
    return result


@router.post("/presentations/generate/stream")
def generatePresentation_Stream(
    request: Request,
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received mock stream request:", presentationGenerateRequest)

    result = svc.make_presentation_stream(presentationGenerateRequest)

    return EventSourceResponse(sse_json_by_json(request, result), ping=None)


# Mock endpoints for testing without LLM calls
@router.post("/outline/generate/mock")
def generateOutline_Mock(
    svc: ContentServiceDep, outlineGenerateRequest: OutlineGenerateRequest
) -> str:
    print("Received mock stream request:", outlineGenerateRequest)
    result = svc.make_outline_mock(outlineGenerateRequest)
    return result


@router.post("/outline/generate/stream/mock")
async def generateOutline_Mock_Stream(
    request: Request,
    outlineGenerateRequest: OutlineGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received mock stream request:", outlineGenerateRequest)

    async def event_stream():
        for chunk in svc.make_outline_stream_mock():
            if await request.is_disconnected():
                break
            yield {
                "data": base64.b64encode(chunk.encode("utf-8")).decode("ascii")
            }

    return EventSourceResponse(event_stream(), media_type="text/event-stream")


@router.post("/presentations/generate/mock")
def generatePresentation_Mock(
    svc: ContentServiceDep,
    presentationGenerateRequest: PresentationGenerateRequest,
):
    print("Received mock stream request:", presentationGenerateRequest)
    result = svc.make_presentation_mock(presentationGenerateRequest)
    return result


@router.post("/presentations/generate/stream/mock")
async def generatePresentation_Mock_Stream(
    request: Request,
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received mock stream request:", presentationGenerateRequest)

    async def event_stream():
        async for obj in svc.make_presentation_stream_mock():
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    return EventSourceResponse(event_stream(), media_type="text/event-stream")


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

    return {
        "images": result["images"],
        "count": result["count"],
        "error": None,
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

    return {
        "images": result["images"],
        "count": result["count"],
        "error": None,
    }


@router.post("/mindmap/generate")
def generateMindmap(
    mindmapGenerateRequest: MindmapGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received mindmap generation request:", mindmapGenerateRequest)
    result = svc.generate_mindmap(mindmapGenerateRequest)
    return result


@router.post("/mindmap/generate/mock")
def generateMindmap_Mock(
    svc: ContentServiceDep,
    mindmapGenerateRequest: MindmapGenerateRequest,
):
    print("Received mock mindmap generation request:", mindmapGenerateRequest)
    result = svc.generate_mindmap_mock(mindmapGenerateRequest)
    return result
