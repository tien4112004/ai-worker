import functools

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)


def trace_span(
    span_name: str,
    system: str,
    model_attr: str = "model",
    input_arg: str = "message",
):
    """Decorator that wraps a method in an OpenInference LLM span.

    Handles span lifecycle, common attributes, and status codes.
    The decorated method can still set additional attributes (e.g. token
    counts) via ``trace.get_current_span()``.

    Error convention: if the return value is a dict with an ``"error"`` key
    the span status is set to ERROR.  Unhandled exceptions are recorded on
    the span and re-raised.

    Args:
        span_name: Span name shown in Phoenix.
        system: LLM provider identifier (e.g. "google", "openai").
        model_attr: Attribute name on ``self`` that holds the model identifier.
        input_arg: Parameter name of the input prompt/message.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            model_name = getattr(self, model_attr)
            message = kwargs.get(input_arg, args[0] if args else "")

            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM"
                )
                span.set_attribute(SpanAttributes.LLM_SYSTEM, system)
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
                span.set_attribute(SpanAttributes.INPUT_VALUE, message)

                try:
                    result = func(self, *args, **kwargs)

                    if isinstance(result, dict) and "error" in result:
                        span.set_status(
                            Status(
                                StatusCode.ERROR, description=result["error"]
                            )
                        )
                    else:
                        span.set_status(Status(StatusCode.OK))

                    return result
                except Exception as e:
                    span.set_status(
                        Status(StatusCode.ERROR, description=str(e))
                    )
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator
