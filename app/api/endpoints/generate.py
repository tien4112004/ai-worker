from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, StreamingResponse
from httpcore import Response

from app.depends import ContentServiceDep
from app.schemas.image_content import (
    ImageGenerateRequest,
    ImageGenerateResponse,
)
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

    return StreamingResponse(
        sse_word_by_word(request, result), media_type="text/event-stream"
    )


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
    result = svc.make_presentation_stream(presentationGenerateRequest)

    return StreamingResponse(
        sse_json_by_json(request, result), media_type="text/event-stream"
    )


# Mock endpoints for testing without LLM calls
@router.post("/outline/generate/mock")
def generateOutline_Mock(
    svc: ContentServiceDep, outlineGenerateRequest: OutlineGenerateRequest
) -> str:
    print("Received mock stream request:", outlineGenerateRequest)
    result = svc.make_outline_mock(outlineGenerateRequest)
    return result


@router.post("/outline/generate/stream/mock")
def generateOutline_Mock_Stream(
    request: Request,
    outlineGenerateRequest: OutlineGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received mock stream request:", outlineGenerateRequest)
    result = svc.make_outline_stream_mock()

    return StreamingResponse(
        sse_word_by_word(request, result), media_type="text/event-stream"
    )


@router.post("/presentations/generate/mock")
def generatePresentation_Mock(
    svc: ContentServiceDep,
    presentationGenerateRequest: PresentationGenerateRequest,
):
    print("Received mock stream request:", presentationGenerateRequest)
    result = svc.make_presentation_mock(presentationGenerateRequest)
    return result


@router.post("/presentations/generate/stream/mock")
def generatePresentation_Mock_Stream(
    request: Request,
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentServiceDep,
):
    print("Received mock stream request:", presentationGenerateRequest)
    result = svc.make_presentation_stream_mock()
    return StreamingResponse(
        sse_json_by_json(request, result), media_type="text/event-stream"
    )


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

    return {"base64_image": result["base64_image"]}
