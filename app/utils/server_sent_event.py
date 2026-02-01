import base64
import json
import re
from typing import Any, Generator, Iterable, List, Optional, Tuple

from app.schemas.token_usage import TokenUsage


# VIBE CODE
async def sse_word_by_word(
    request, chunks: Iterable, token_usage: Optional[Any] = None
):
    print("Starting SSE word by word")
    if await request.is_disconnected():
        print("Client disconnected")
        return

    buffer = ""

    try:
        # Send content chunks
        for chunk in chunks:
            if isinstance(chunk, TokenUsage):
                token_usage = chunk
                continue

            if await request.is_disconnected():
                print("Client disconnected during streaming")
                return

            if chunk:
                buffer += str(chunk)

                # Split on whitespace, keeping separators as elements:
                # "### Âm Thanh" -> ['###', ' ', 'Âm', ' ', 'Thanh']
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
                    json.dumps(
                        {
                            "token_usage": {
                                "input_tokens": token_usage.input_tokens,
                                "output_tokens": token_usage.output_tokens,
                                "total_tokens": token_usage.total_tokens,
                                "model": token_usage.model,
                                "provider": token_usage.provider,
                            }
                        }
                    ).encode("utf-8")
                ).decode("ascii")
            }

    except Exception as e:
        print(f"Error in word-by-word streaming: {e}")
        error_encoded = base64.b64encode(
            f"Error: {str(e)}".encode("utf-8")
        ).decode("ascii")
        yield {"data": error_encoded}


def _find_complete_json_object(buffer: str) -> Optional[Tuple[str, int]]:
    """Find a complete JSON object in the buffer.

    Returns:
        Tuple of (json_str, end_idx) if found, None otherwise
    """
    start_idx = buffer.find("{")
    if start_idx == -1:
        return None

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

    if brace_count == 0:
        json_str = buffer[start_idx : end_idx + 1].strip()
        return (json_str, end_idx)

    return None


def _process_json_object(
    json_str: str, buffer: str, start_idx: int, end_idx: int
) -> Tuple[Optional[str], str]:
    """Process a JSON object and return SSE event data and updated buffer.

    Returns:
        Tuple of (event_data, updated_buffer)
    """
    try:
        json_obj = json.loads(json_str)

        # Check if it's a valid object with type field
        if isinstance(json_obj, dict) and "type" in json_obj:
            event_data = (
                f"data: {json.dumps(json_obj, ensure_ascii=False)}\n\n"
            )
            updated_buffer = buffer[end_idx + 1 :].lstrip()
            return (event_data, updated_buffer)

        # Valid JSON but no type field, just remove it from buffer
        return (None, buffer[end_idx + 1 :].lstrip())

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        # Remove the problematic part and continue
        return (None, buffer[start_idx + 1 :])


def _create_token_usage_event(token_usage: TokenUsage) -> str:
    """Create SSE event data for token usage."""
    return f"data: {json.dumps({'token_usage': {'input_tokens': token_usage.input_tokens, 'output_tokens': token_usage.output_tokens, 'total_tokens': token_usage.total_tokens, 'model': token_usage.model, 'provider': token_usage.provider}}, ensure_ascii=False)}\n\n"


# VIBE CODE
async def sse_json_by_json(
    request, chunks: Iterable, token_usage: Optional[Any] = None
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
            if isinstance(chunk, TokenUsage):
                token_usage = chunk
                continue

            if await request.is_disconnected():
                print("Client disconnected during streaming")
                return

            if chunk:
                buffer += chunk
                buffer = buffer.replace("```json", "").replace("```", "")
                print(f"Current buffer: {buffer}")

                # Look for and process complete JSON objects in the buffer
                while True:
                    result = _find_complete_json_object(buffer)
                    if result is None:
                        break

                    json_str, end_idx = result
                    start_idx = buffer.find("{")

                    event_data, buffer = _process_json_object(
                        json_str, buffer, start_idx, end_idx
                    )

                    if event_data:
                        yield event_data

        # Send token usage as final event
        if token_usage:
            yield _create_token_usage_event(token_usage)

    except Exception as e:
        print(f"Error in SSE streaming: {e}")
        yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'}, ensure_ascii=False)}\n\n"
