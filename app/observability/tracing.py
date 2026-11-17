# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry distributed tracing setup with Google Cloud Trace export.

Links spans end-to-end from incoming user query through coordinator routing,
parallel specialist discovery, LLM reasoning, tool execution, and async memory
consolidation.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import os
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.security.pii_redactor import get_pii_redactor

F = TypeVar("F", bound=Callable[..., Any])

_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "travel-concierge-agent")
_IN_MEMORY_EXPORTER: InMemorySpanExporter | None = None
_TRACING_INITIALIZED = False


def configure_distributed_tracing(
    *,
    service_name: str = _SERVICE_NAME,
    enable_in_memory_exporter: bool = False,
) -> TracerProvider:
    """Configures the OpenTelemetry `TracerProvider` and Cloud Trace exporter.

    Safe to invoke multiple times; preserves an already-configured SDK provider
    (for example when `get_fast_api_app(otel_to_cloud=True)` initializes Cloud
    Trace first) while attaching optional test exporters if requested.

    Args:
        service_name: Logical OpenTelemetry `service.name` resource attribute.
        enable_in_memory_exporter: Attaches an `InMemorySpanExporter` for
            deterministic span inspection in automated tests.

    Returns:
        The active OpenTelemetry `TracerProvider`.
    """
    global _IN_MEMORY_EXPORTER, _TRACING_INITIALIZED

    existing_provider = trace.get_tracer_provider()
    if isinstance(existing_provider, TracerProvider):
        provider = existing_provider
    else:
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": os.environ.get("AGENT_VERSION", "0.1.0"),
            }
        )
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

    if enable_in_memory_exporter and _IN_MEMORY_EXPORTER is None:
        _IN_MEMORY_EXPORTER = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_IN_MEMORY_EXPORTER))

    if not _TRACING_INITIALIZED:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        export_to_cloud = os.environ.get(
            "OTEL_TRACES_EXPORTER", "auto"
        ).lower() not in {"none", "memory"}
        if (
            export_to_cloud
            and project_id
            and not project_id.startswith("your-")
            and os.environ.get("PYTEST_CURRENT_TEST") is None
        ):
            try:
                from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

                cloud_exporter = CloudTraceSpanExporter(project_id=project_id)
                provider.add_span_processor(BatchSpanProcessor(cloud_exporter))
            except Exception:
                # Fall back cleanly when running offline without GCP credentials.
                pass
        _TRACING_INITIALIZED = True

    return provider


def get_in_memory_span_exporter() -> InMemorySpanExporter | None:
    """Returns the active `InMemorySpanExporter` when enabled for tests."""
    return _IN_MEMORY_EXPORTER


def get_tracer(instrumenting_module_name: str = "travel_concierge") -> trace.Tracer:
    """Returns an OpenTelemetry tracer bound to `instrumenting_module_name`."""
    configure_distributed_tracing()
    return trace.get_tracer(instrumenting_module_name)


def get_current_trace_identifiers() -> tuple[str | None, str | None, bool]:
    """Extracts `(trace_id_hex, span_id_hex, trace_sampled)` from active context."""
    span = trace.get_current_span()
    span_ctx = span.get_span_context()
    if not span_ctx.is_valid:
        return None, None, False
    trace_id = f"{span_ctx.trace_id:032x}"
    span_id = f"{span_ctx.span_id:016x}"
    sampled = bool(span_ctx.trace_flags.sampled)
    return trace_id, span_id, sampled


@contextlib.contextmanager
def start_sanitized_span(
    span_name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Iterator[trace.Span]:
    """Starts a child span and attaches PII-scrubbed attributes."""
    tracer = get_tracer()
    redactor = get_pii_redactor()
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            clean_attrs = redactor.redact_data(attributes)
            for key, value in clean_attrs.items():
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(key, value)
                elif value is not None:
                    span.set_attribute(key, str(value))
        yield span


def traced_span(span_name: str, **static_attributes: Any) -> Callable[[F], F]:
    """Decorator that wraps sync or async callables in a linked OTel span."""

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                attrs = {"code.function": func.__qualname__, **static_attributes}
                with start_sanitized_span(span_name, attributes=attrs) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("operation.status", "ok")
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_attribute("operation.status", "error")
                        raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            attrs = {"code.function": func.__qualname__, **static_attributes}
            with start_sanitized_span(span_name, attributes=attrs) as span:
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, dict) and "status" in result:
                        span.set_attribute("operation.status", str(result["status"]))
                    else:
                        span.set_attribute("operation.status", "ok")
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_attribute("operation.status", "error")
                    raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator
