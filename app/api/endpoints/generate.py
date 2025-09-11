from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.depends import ContentServiceDep
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)
from app.utils.server_side_event import sse_json_by_json, sse_word_by_word

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
    result = svc.make_outline_mock()
    return result


@router.post("/outline/generate/stream/mock")
def generateOutline_Mock_Stream(
    request: Request,
    outlineGenerateRequest: OutlineGenerateRequest,
    svc: ContentServiceDep,
):
    result = svc.make_outline_stream_mock()

    return StreamingResponse(
        sse_word_by_word(request, result), media_type="text/event-stream"
    )


@router.post("/presentations/generate/mock")
def generatePresentation_Mock(
    svc: ContentServiceDep,
    presentationGenerateRequest: PresentationGenerateRequest,
):
    result = svc.make_presentation_mock()
    return result


@router.post("/presentations/generate/stream/mock")
def generatePresentation_Mock_Stream(
    request: Request,
    presentationGenerateRequest: PresentationGenerateRequest,
    svc: ContentServiceDep,
):
    result = svc.make_presentation_stream_mock()
    print("Result from mock stream:", result)
    return StreamingResponse(
        sse_json_by_json(request, result), media_type="text/event-stream"
    )
