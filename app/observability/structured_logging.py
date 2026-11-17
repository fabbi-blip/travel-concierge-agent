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

"""Structured JSON logging with Cloud Trace correlation and active PII scrubbing.

Replaces unstructured `print()` calls across the codebase with single-line JSON
records conforming to Google Cloud Logging's structured payload specification
(`severity`, `message`, `timestamp`, `logging.googleapis.com/trace`,
`logging.googleapis.com/spanId`, and rich domain metadata).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from typing import Any

from app.observability.tracing import get_current_trace_identifiers
from app.security.pii_redactor import PiiRedactionService, get_pii_redactor

_STANDARD_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class PiiRedactingLogFilter(logging.Filter):
    """Logging filter that scrubs PII from messages and structured extra fields."""

    def __init__(self, redactor: PiiRedactionService | None = None) -> None:
        super().__init__(name="pii_redacting_filter")
        self._redactor = redactor or get_pii_redactor()

    def filter(self, record: logging.LogRecord) -> bool:
        raw_message = record.getMessage()
        record.msg = self._redactor.redact_text(raw_message)
        record.args = ()

        for key in list(record.__dict__.keys()):
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_"):
                record.__dict__[key] = self._redactor.redact_data(record.__dict__[key])
        return True


class StructuredJsonFormatter(logging.Formatter):
    """Serializes `LogRecord` instances as Cloud Logging structured JSON."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        service_name: str | None = None,
        redactor: PiiRedactionService | None = None,
    ) -> None:
        super().__init__()
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._service_name = service_name or os.environ.get(
            "OTEL_SERVICE_NAME", "travel-concierge-agent"
        )
        self._redactor = redactor or get_pii_redactor()

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.UTC
        ).isoformat()

        message = self._redactor.redact_text(record.getMessage())
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "severity": record.levelname,
            "logger": record.name,
            "service": self._service_name,
            "message": message,
        }

        trace_id, span_id, sampled = get_current_trace_identifiers()
        if trace_id:
            if self._project_id and not self._project_id.startswith("your-"):
                payload["logging.googleapis.com/trace"] = (
                    f"projects/{self._project_id}/traces/{trace_id}"
                )
            else:
                payload["trace_id"] = trace_id
        if span_id:
            payload["logging.googleapis.com/spanId"] = span_id
        if trace_id:
            payload["logging.googleapis.com/trace_sampled"] = sampled

        metadata: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_"):
                metadata[key] = self._redactor.redact_data(value)

        if metadata:
            payload.update(metadata)

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            payload["exception"] = self._redactor.redact_text(exc_text)

        return json.dumps(payload, default=str, ensure_ascii=False)


_CONFIGURED = False


def configure_structured_logging(level: int | str | None = None) -> logging.Logger:
    """Configures structured JSON logging with PII redaction for the `app` tree."""
    global _CONFIGURED
    resolved_level = level or os.environ.get("LOG_LEVEL", "INFO").upper()

    app_logger = logging.getLogger("app")
    app_logger.setLevel(resolved_level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        handler.addFilter(PiiRedactingLogFilter())

        # Remove duplicate handlers if re-imported while preserving pytest caplog.
        app_logger.handlers = [handler]
        app_logger.propagate = True
        _CONFIGURED = True

    return app_logger


def get_structured_logger(name: str) -> logging.Logger:
    """Returns a child logger under the structured `app` hierarchy."""
    configure_structured_logging()
    normalized = name if name.startswith("app.") or name == "app" else f"app.{name}"
    return logging.getLogger(normalized)
