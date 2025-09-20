import base64
import json
import re
from typing import Any, Generator


async def sse_word_by_word(request, generator: Generator[Any, Any, None]):
    print("Starting SSE word by word")
    if await request.is_disconnected():
        print("Client disconnected")
        return
    # Each chunk is a statement, split it word by word
    for chunk in generator:
        if chunk != "":
            encoded = str(base64.b64encode(str(chunk).encode("utf-8")).decode("ascii"))
            yield {"data": encoded}


async def sse_json_by_json(request, generator: Generator[Any, Any, None]):
    """Stream JSON objects one at a time in SSE format."""
    print("Starting SSE JSON by JSON streaming")

    if await request.is_disconnected():
        print("Client disconnected")
        return

    buffer = ""

    try:
        for chunk in generator:
            if await request.is_disconnected():
                print("Client disconnected during streaming")
                return

            if chunk:
                buffer += chunk

                # Look for complete JSON objects in the buffer
                # Pattern to match JSON objects between { and }
                json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

                while True:
                    match = re.search(json_pattern, buffer)
                    if not match:
                        break

                    json_str = match.group(0)

                    try:
                        # Validate that it's proper JSON
                        json_obj = json.loads(json_str)

                        # Check if it has the expected slide structure
                        if isinstance(json_obj, dict) and "title" in json_obj:
                            # Send as SSE event
                            yield f"data: {json.dumps(json_obj, ensure_ascii=False)}\n\n"

                        # Remove the processed JSON from buffer
                        buffer = buffer[match.end() :]

                    except json.JSONDecodeError:
                        # If JSON is invalid, move past this match
                        buffer = buffer[match.start() + 1 :]
                        break

    except Exception as e:
        print(f"Error in SSE streaming: {e}")
        yield f'{{"error": "Streaming error: {str(e)}"}}\n\n'
