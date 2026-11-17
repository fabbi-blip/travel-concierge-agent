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

"""Unit tests for PII Redaction (Cloud DLP + regex) and Secure Secret Management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.security.pii_redactor import PiiRedactionService
from app.security.secrets import SecretManagerService, SecretResolutionError


@dataclass
class _MockDlpItem:
    value: str


@dataclass
class _MockDlpResponse:
    item: _MockDlpItem


class _FakeCloudDlpClient:
    """In-memory test double for `google.cloud.dlp_v2.DlpServiceClient`."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def deidentify_content(self, request: dict[str, Any]) -> _MockDlpResponse:
        self.requests.append(request)
        raw_val = str(request["item"]["value"])
        return _MockDlpResponse(
            item=_MockDlpItem(value=raw_val.replace("John Doe", "[PERSON_NAME]"))
        )


def test_pii_redactor_scrubs_sensitive_tokens_and_cloud_dlp() -> None:
    """Verifies regex + Cloud DLP scrubbing across strings and nested dicts."""
    fake_dlp = _FakeCloudDlpClient()
    redactor = PiiRedactionService(
        project_id="demo-travel-sec",
        enable_cloud_dlp=True,
        dlp_client=fake_dlp,
    )

    raw_note = (
        "Traveler John Doe (email: john.doe@example.com, phone: +1-415-555-0199, "
        "SSN 123-45-6789, Passport No: X98765432) paid with card 4111-2222-3333-4444."
    )
    scrubbed = redactor.redact_text(raw_note)

    assert "john.doe@example.com" not in scrubbed
    assert "415-555-0199" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "X98765432" not in scrubbed
    assert "4111-2222-3333-4444" not in scrubbed
    assert "John Doe" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PAYMENT_CARD]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "[REDACTED_PASSPORT]" in scrubbed
    assert "[PERSON_NAME]" in scrubbed
    assert len(fake_dlp.requests) == 1

    nested = {
        "api_key": "sk-secretvalue123456789012345",
        "receipt": {
            "merchant": "Shinjuku Hotel",
            "notes": "Contacted billing@hotel.jp",
        },
    }
    clean_nested = redactor.redact_data(nested)
    assert clean_nested["api_key"] == "[REDACTED_SENSITIVE_FIELD]"
    assert clean_nested["receipt"]["notes"] == "Contacted [REDACTED_EMAIL]"


@dataclass
class _FakeSecretPayload:
    data: bytes


@dataclass
class _FakeSecretVersionResponse:
    payload: _FakeSecretPayload


class _FakeSecretManagerClient:
    """Test double verifying versioned Secret Manager resource resolution."""

    def __init__(self, secrets: dict[str, str]) -> None:
        self._secrets = secrets
        self.calls: list[str] = []

    def access_secret_version(
        self, request: dict[str, Any]
    ) -> _FakeSecretVersionResponse:
        name = str(request["name"])
        self.calls.append(name)
        if name not in self._secrets:
            raise KeyError(f"Secret not found: {name}")
        return _FakeSecretVersionResponse(
            payload=_FakeSecretPayload(data=self._secrets[name].encode("utf-8"))
        )


def test_secret_manager_resolves_and_caches_secrets() -> None:
    """Verifies versioned Secret Manager retrieval, TTL caching, and fail-closed errors."""
    resource = "projects/prod-travel-01/secrets/partner-api-key/versions/latest"
    fake_client = _FakeSecretManagerClient({resource: "resolved-from-gsm-v1"})
    service = SecretManagerService(
        project_id="prod-travel-01",
        client=fake_client,
        cache_ttl_seconds=120,
        allow_env_fallback=False,
    )

    first = service.get_secret("partner-api-key", required=True)
    second = service.get_secret("partner-api-key", required=True)
    assert first == "resolved-from-gsm-v1"
    assert second == "resolved-from-gsm-v1"
    assert fake_client.calls == [resource]

    with pytest.raises(SecretResolutionError):
        service.get_secret("nonexistent-secret", required=True)


def test_repository_contains_zero_hardcoded_api_keys() -> None:
    """Scans tracked Python and Terraform files to guarantee zero hardcoded keys."""
    repo_root = Path(__file__).resolve().parents[2]
    checked_files = 0
    for ext in ("*.py", "*.tf", "*.yaml", "*.yml"):
        for path in repo_root.rglob(ext):
            if ".venv" in path.parts or "tests" in path.parts:
                continue
            checked_files += 1
            content = path.read_text(encoding="utf-8")
            for token in content.split():
                cleaned = token.strip("\"'(),;[]{}")
                assert not SecretManagerService.looks_like_raw_credential(cleaned), (
                    f"Potential hardcoded credential detected in {path}"
                )
    assert checked_files >= 15
