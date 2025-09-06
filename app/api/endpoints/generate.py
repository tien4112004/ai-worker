import asyncio
from collections.abc import Iterator
from fastapi import APIRouter, Depends
from typing import Annotated

from fastapi.responses import StreamingResponse
from app.core.depends import ContentServiceDep, LLMServiceDep

from requests import Request, get, request
from app.core.depends import get_content_service
from app.schemas.slide_content import OutlineGenerateRequest, OutlineGenerateResponse
from app.services.content_service import ContentService

router = APIRouter(tags=["generate"], dependencies=[Depends(get_content_service)])

@router.post("/outline/generate")
def generateOutline(payload: OutlineGenerateRequest, svc: ContentServiceDep):
    result = svc.make_outline(payload)
    return result

async def sse(generator: str):
    # for aiMessageChunk in generator:
        # Send each word as a separate SSE event
    content = generator.split(' ')
    for word in content:
        if word != '' and word.strip():
            yield f"{word} "
        await asyncio.sleep(0.1)  # Delay between words

@router.post("/outline/generate/stream")
def generateOutline_Stream(payload: OutlineGenerateRequest, svc: Annotated[ContentService, Depends(get_content_service)]):
    result = svc.make_outline(payload)

    return StreamingResponse(sse(result), media_type="text/text_plain")
