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

"""Active PII scrubbing pipeline for structured logging and memory persistence.

Combines deterministic regex scrubbing with optional Google Cloud Sensitive
Data Protection (Cloud DLP API) de-identification so sensitive traveler data
(payment cards, SSNs, passport numbers, emails, phone numbers, and auth tokens)
is redacted prior to storage.
"""

from __future__ import annotations

import functools
import importlib
import os
import re
from typing import Any, Protocol


class DlpClientProtocol(Protocol):
    """Protocol for Google Cloud DLP clients (`google.cloud.dlp_v2.DlpServiceClient`)."""

    def deidentify_content(self, request: dict[str, Any]) -> Any:
        """De-identifies sensitive infoTypes in text content."""


_SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "db_pass",
        "secret",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "authorization",
        "card_number",
        "credit_card",
        "cvv",
        "cvc",
        "ssn",
        "social_security_number",
        "passport_number",
    }
)

# Ordered from most specific patterns to broadest to avoid partial-match collisions.
_REGEX_SCRUB_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Bearer tokens & Google/OpenAI style API keys
    (
        re.compile(
            r"(?i)\b(?:bearer\s+[A-Za-z0-9\-._~+/]+=*|AIza[0-9A-Za-z\-_]{35}|sk-[A-Za-z0-9]{20,})\b"
        ),
        "[REDACTED_AUTH_TOKEN]",
    ),
    # Payment card numbers (13-19 digits with optional spaces or hyphens)
    (
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{1,7}\b"),
        "[REDACTED_PAYMENT_CARD]",
    ),
    # US Social Security Numbers (XXX-XX-XXXX)
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    # Passport identifiers when preceded by passport context or standard 9-char format
    (
        re.compile(r"(?i)(passport(?:\s*(?:no|number|id|#))?[:\s]+)([A-Z0-9]{6,9})\b"),
        r"\1[REDACTED_PASSPORT]",
    ),
    # Email addresses
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    # International & North American phone numbers (avoiding ISO dates like 2026-11-15)
    (
        re.compile(
            r"(?<!\d)(?:\+\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
        ),
        "[REDACTED_PHONE]",
    ),
)

_CLOUD_DLP_INFO_TYPES: tuple[dict[str, str], ...] = (
    {"name": "EMAIL_ADDRESS"},
    {"name": "PHONE_NUMBER"},
    {"name": "CREDIT_CARD_NUMBER"},
    {"name": "US_SOCIAL_SECURITY_NUMBER"},
    {"name": "PASSPORT"},
    {"name": "AUTH_TOKEN"},
)


class PiiRedactionService:
    """Scrubs PII from strings, dictionaries, lists, and event payloads.

    Executes a fast deterministic regex pass on every payload and optionally
    invokes Google Cloud Sensitive Data Protection (`dlp_v2`) for deep contextual
    entity redaction when enabled in cloud environments.
    """

    def __init__(
        self,
        project_id: str | None = None,
        enable_cloud_dlp: bool | None = None,
        dlp_client: DlpClientProtocol | None = None,
    ) -> None:
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if enable_cloud_dlp is None:
            self._enable_cloud_dlp = os.environ.get(
                "ENABLE_CLOUD_DLP", "false"
            ).lower() in {"1", "true", "yes"}
        else:
            self._enable_cloud_dlp = enable_cloud_dlp
        self._dlp_client = dlp_client

    def _resolve_dlp_client(self) -> DlpClientProtocol | None:
        if self._dlp_client is not None:
            return self._dlp_client
        if not self._enable_cloud_dlp or not self._project_id:
            return None
        try:
            dlp_mod = importlib.import_module("google.cloud.dlp_v2")
            self._dlp_client = dlp_mod.DlpServiceClient()
            return self._dlp_client
        except Exception:
            self._enable_cloud_dlp = False
            return None

    def redact_text(self, text: str) -> str:
        """Redacts PII tokens from a string using regex rules and optional Cloud DLP.

        Args:
            text: Raw input text that may contain sensitive personal identifiers.

        Returns:
            Sanitized string safe for structured logs and long-term memory storage.
        """
        if not text:
            return text

        scrubbed = text
        for pattern, replacement in _REGEX_SCRUB_RULES:
            scrubbed = pattern.sub(replacement, scrubbed)

        client = self._resolve_dlp_client()
        if client is not None and self._project_id and scrubbed.strip():
            try:
                parent = f"projects/{self._project_id}/locations/global"
                request = {
                    "parent": parent,
                    "inspect_config": {"info_types": list(_CLOUD_DLP_INFO_TYPES)},
                    "deidentify_config": {
                        "info_type_transformations": {
                            "transformations": [
                                {
                                    "primitive_transformation": {
                                        "replace_with_info_type_config": {}
                                    }
                                }
                            ]
                        }
                    },
                    "item": {"value": scrubbed},
                }
                response = client.deidentify_content(request=request)
                if getattr(response, "item", None) and getattr(
                    response.item, "value", None
                ):
                    scrubbed = str(response.item.value)
            except Exception:
                # Regex redaction has already run; never fail open with raw text.
                pass

        return scrubbed

    def redact_data(self, payload: Any, *, max_depth: int = 8) -> Any:
        """Recursively scrubs sensitive keys and string values across nested data.

        Args:
            payload: Arbitrary dictionary, list, tuple, or primitive value.
            max_depth: Maximum recursion depth to guard against cyclic structures.

        Returns:
            A sanitized copy of `payload` with all PII replaced by redaction tags.
        """
        if max_depth <= 0:
            return "[TRUNCATED_DEPTH]"

        if isinstance(payload, str):
            return self.redact_text(payload)

        if isinstance(payload, dict):
            sanitized: dict[str, Any] = {}
            for key, value in payload.items():
                normalized_key = str(key).lower().strip()
                if normalized_key in _SENSITIVE_FIELD_NAMES:
                    sanitized[str(key)] = "[REDACTED_SENSITIVE_FIELD]"
                else:
                    sanitized[str(key)] = self.redact_data(
                        value, max_depth=max_depth - 1
                    )
            return sanitized

        if isinstance(payload, list):
            return [self.redact_data(item, max_depth=max_depth - 1) for item in payload]

        if isinstance(payload, tuple):
            return tuple(
                self.redact_data(item, max_depth=max_depth - 1) for item in payload
            )

        return payload

    def contains_pii(self, text: str) -> bool:
        """Returns True if deterministic PII patterns are detected in `text`."""
        if not text:
            return False
        return self.redact_text(text) != text


@functools.lru_cache(maxsize=1)
def get_pii_redactor() -> PiiRedactionService:
    """Returns the process-wide `PiiRedactionService` instance."""
    return PiiRedactionService()
