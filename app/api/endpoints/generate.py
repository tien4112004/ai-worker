import asyncio
from collections.abc import Iterator
from typing import Annotated, Any, Generator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, BaseMessageChunk

from app.depends import ContentServiceDep
from app.schemas.slide_content import OutlineGenerateRequest
from app.services.content_service import ContentService

router = APIRouter(tags=["generate"])


@router.post("/outline/generate")
def generateOutline(payload: OutlineGenerateRequest, svc: ContentServiceDep):
    result = svc.make_outline(payload)
    return result


async def sse(request, generator: Generator[Any, Any, None]):
    if await request.is_disconnected():
        print("Client disconnected")
        return
    for chunk in generator:
        content = chunk.content
        if content != "":
            yield f"{content} "
        await asyncio.sleep(0.1)  # Delay between words


@router.post("/outline/generate/stream")
def generateOutline_Stream(
    request: Request, payload: OutlineGenerateRequest, svc: ContentServiceDep
):
    result = svc.make_outline_stream(payload)

    return StreamingResponse(
        sse(request, result), media_type="text/event-stream"
    )
