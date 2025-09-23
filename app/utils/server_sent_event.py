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
            encoded = str(
                base64.b64encode(str(chunk).encode("utf-8")).decode("ascii")
            )
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

                # Remove code block markers if present
                buffer = buffer.replace("```json", "").replace("```", "")

                print(f"Current buffer: {buffer}")

                # Look for complete JSON objects in the buffer
                # Pattern to match JSON objects that start with { and end with }
                # Using a more robust approach to find balanced braces
                while True:
                    # Find the start of a JSON object
                    start_idx = buffer.find("{")
                    if start_idx == -1:
                        break

                    # Count braces to find the complete JSON object
                    brace_count = 0
                    end_idx = start_idx

                    for i in range(start_idx, len(buffer)):
                        if buffer[i] == "{":
                            brace_count += 1
                        elif buffer[i] == "}":
                            brace_count -= 1

                        if brace_count == 0:
                            end_idx = i
                            break

                    # If we found a complete JSON object
                    if brace_count == 0:
                        json_str = buffer[start_idx : end_idx + 1].strip()

                        try:
                            # Validate that it's proper JSON
                            json_obj = json.loads(json_str)

                            # Check if it's a valid object with type field
                            if (
                                isinstance(json_obj, dict)
                                and "type" in json_obj
                            ):
                                # Send as SSE event
                                yield f"data: {json.dumps(json_obj, ensure_ascii=False)}\n\n"

                            # Remove the processed JSON from buffer
                            buffer = buffer[end_idx + 1 :].lstrip()

                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            # Remove the problematic part and continue
                            buffer = buffer[start_idx + 1 :]
                    else:
                        # Incomplete JSON object, wait for more data
                        break

    except Exception as e:
        print(f"Error in SSE streaming: {e}")
        yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'}, ensure_ascii=False)}\n\n"
