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

"""Secure secret resolution backed by Google Cloud Secret Manager.

Eliminates hardcoded API keys and database passwords by fetching versioned
secrets via Application Default Credentials (ADC) and IAM Workload Identity.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import threading
import time
from typing import Any, Protocol

from google.api_core import exceptions as gcp_exceptions
from google.cloud import secretmanager

logger = logging.getLogger(__name__)

_SUSPICIOUS_LITERAL_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^AIza[0-9A-Za-z\-_]{35}$"),
    re.compile(r"^sk-[A-Za-z0-9]{20,}$"),
    re.compile(r"^ya29\.[0-9A-Za-z\-_]+$"),
)


class SecretManagerClientProtocol(Protocol):
    """Protocol for dependency-injected Secret Manager clients in tests."""

    def access_secret_version(self, request: dict[str, Any]) -> Any:
        """Accesses a secret version payload from Google Cloud Secret Manager."""


class SecretResolutionError(RuntimeError):
    """Raised when a required secret cannot be securely retrieved."""


class SecretManagerService:
    """Thread-safe Google Cloud Secret Manager wrapper with TTL caching.

    All external partner credentials, database passwords, and API tokens are
    resolved dynamically from Secret Manager rather than embedded in source
    code. Local offline unit testing can opt into environment-variable
    injection without persisting secrets on disk.
    """

    def __init__(
        self,
        project_id: str | None = None,
        client: SecretManagerClientProtocol | None = None,
        cache_ttl_seconds: int = 300,
        allow_env_fallback: bool | None = None,
    ) -> None:
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._client = client
        self._cache_ttl_seconds = max(0, cache_ttl_seconds)
        if allow_env_fallback is None:
            self._allow_env_fallback = os.environ.get(
                "ALLOW_ENV_SECRET_FALLBACK", "true"
            ).lower() in {"1", "true", "yes"}
        else:
            self._allow_env_fallback = allow_env_fallback
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}
        self._lock = threading.Lock()

    def _get_client(self) -> SecretManagerClientProtocol:
        if self._client is None:
            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get_secret(
        self,
        secret_id: str,
        *,
        version: str = "latest",
        env_var_name: str | None = None,
        required: bool = False,
        default: str | None = None,
    ) -> str | None:
        """Retrieves a secret value from Google Cloud Secret Manager.

        Resolution priority:
        1. Cloud Run / Agent Runtime Secret Manager volume or env mount if
           explicitly injected by runtime infrastructure (`env_var_name`).
        2. Direct Google Cloud Secret Manager API call via IAM / ADC.
        3. Optional default value when `required=False`.

        Args:
            secret_id: Identifier of the secret in Google Cloud Secret Manager.
            version: Secret version to access (defaults to `"latest"`).
            env_var_name: Optional runtime environment variable bound via
                Cloud Run `secret_key_ref`.
            required: Whether to raise `SecretResolutionError` if unavailable.
            default: Fallback value when `required=False` and resolution fails.

        Returns:
            The decoded secret string, or `default` if optional and absent.

        Raises:
            SecretResolutionError: If `required=True` and the secret cannot be
                resolved from Secret Manager or an authorized runtime binding.
        """
        cache_key = (secret_id, version)
        now = time.monotonic()

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None and cached[1] > now:
                return cached[0]

        if self._allow_env_fallback and env_var_name:
            env_val = os.environ.get(env_var_name)
            if env_val and not env_val.startswith("your-"):
                self._store_cache(cache_key, env_val, now)
                return env_val

        secret_mgr_enabled = (
            self._client is not None
            or bool(os.environ.get("K_SERVICE"))
            or os.environ.get("ENABLE_SECRET_MANAGER", "false").lower()
            in {"1", "true", "yes"}
        )
        if (
            secret_mgr_enabled
            and self._project_id
            and not self._project_id.startswith("your-")
        ):
            resource_name = (
                f"projects/{self._project_id}/secrets/{secret_id}/versions/{version}"
            )
            try:
                response = self._get_client().access_secret_version(
                    request={"name": resource_name}
                )
                secret_value: str = response.payload.data.decode("utf-8").strip()
                self._store_cache(cache_key, secret_value, now)
                logger.info(
                    "Resolved secret from Secret Manager",
                    extra={
                        "event_type": "SECRET_ACCESS",
                        "secret_id": secret_id,
                        "secret_version": version,
                    },
                )
                return secret_value
            except (gcp_exceptions.GoogleAPICallError, Exception) as exc:
                logger.warning(
                    "Secret Manager lookup failed; falling back if permitted",
                    extra={
                        "event_type": "SECRET_ACCESS_FAILURE",
                        "secret_id": secret_id,
                        "error_type": type(exc).__name__,
                    },
                )

        if required and default is None:
            raise SecretResolutionError(
                f"Required secret '{secret_id}' could not be retrieved from "
                "Google Cloud Secret Manager and no authorized runtime binding was found."
            )
        return default

    def _store_cache(
        self, key: tuple[str, str], value: str, now: float
    ) -> None:
        if self._cache_ttl_seconds <= 0:
            return
        with self._lock:
            self._cache[key] = (value, now + self._cache_ttl_seconds)

    @staticmethod
    def looks_like_raw_credential(candidate: str) -> bool:
        """Detects strings matching common raw API key or bearer token formats."""
        stripped = candidate.strip()
        return any(p.match(stripped) for p in _SUSPICIOUS_LITERAL_KEY_PATTERNS)


@functools.lru_cache(maxsize=1)
def get_secret_manager() -> SecretManagerService:
    """Returns the singleton `SecretManagerService` for the process."""
    return SecretManagerService()
