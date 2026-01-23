import base64
import json
import re
from typing import Any, Generator, List, Optional, Tuple


# VIBE CODE
async def sse_word_by_word(
    request, chunks: List[str], token_usage: Optional[Any] = None
):
    print("Starting SSE word by word")
    if await request.is_disconnected():
        print("Client disconnected")
        return

    buffer = ""

    try:
        # Send content chunks
        for chunk in chunks:
            if await request.is_disconnected():
                print("Client disconnected during streaming")
                return

            if chunk:
                buffer += str(chunk)

                # Split text preserving spaces and newlines
                # This regex splits on word boundaries but keeps the separators
                tokens = re.split(r"(\s+)", buffer)

                # Process all tokens except the last one (which might be incomplete)
                if len(tokens) > 1:
                    complete_tokens = tokens[:-1]
                    buffer = tokens[-1]  # Keep the last token in buffer

                    for token in complete_tokens:
                        if (
                            token
                        ):  # Don't skip empty tokens as they might be important whitespace
                            encoded = base64.b64encode(
                                token.encode("utf-8")
                            ).decode("ascii")
                            yield {"data": encoded}

        # Yield any remaining content in buffer
        if buffer:
            encoded = base64.b64encode(buffer.encode("utf-8")).decode("ascii")
            yield {"data": encoded}
        
        # Send token usage as final event
        if token_usage:
            yield {
                "data": base64.b64encode(
                    json.dumps({
                        "token_usage": {
                            "input_tokens": token_usage.input_tokens,
                            "output_tokens": token_usage.output_tokens,
                            "total_tokens": token_usage.total_tokens,
                            "model": token_usage.model,
                            "provider": token_usage.provider,
                        }
                    }).encode("utf-8")
                ).decode("ascii")
            }

    except Exception as e:
        print(f"Error in word-by-word streaming: {e}")
        error_encoded = base64.b64encode(
            f"Error: {str(e)}".encode("utf-8")
        ).decode("ascii")
        yield {"data": error_encoded}


# VIBE CODE
async def sse_json_by_json(
    request, chunks: List[str], token_usage: Optional[Any] = None
):
    """Stream JSON objects one at a time in SSE format, then send token usage."""
    print("Starting SSE JSON by JSON streaming")

    if await request.is_disconnected():
        print("Client disconnected")
        return

    buffer = ""

    try:
        # Process content chunks
        for chunk in chunks:
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
        
        # Send token usage as final event
        if token_usage:
            yield f"data: {json.dumps({'token_usage': {'input_tokens': token_usage.input_tokens, 'output_tokens': token_usage.output_tokens, 'total_tokens': token_usage.total_tokens, 'model': token_usage.model, 'provider': token_usage.provider}}, ensure_ascii=False)}\n\n"

    except Exception as e:
        print(f"Error in SSE streaming: {e}")
        yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'}, ensure_ascii=False)}\n\n"
