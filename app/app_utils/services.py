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

"""Process-wide ADK session, memory, and artifact services.

Supports persistent conversational history across Cloud SQL (PostgreSQL),
Vertex AI Agent Engine Sessions, and local SQLite (`SqliteSessionService`),
resolving database credentials securely via Google Cloud Secret Manager.
"""

from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.adk.cli.service_registry import get_service_registry
from google.adk.cli.utils.service_factory import create_session_service_from_options
from google.adk.memory import InMemoryMemoryService, VertexAiMemoryBankService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

from app.observability.structured_logging import get_structured_logger
from app.security.secrets import get_secret_manager

logger = get_structured_logger(__name__)

SESSION_SERVICE_URI = "shared://session"
ARTIFACT_SERVICE_URI = "shared://artifact"

_AGENT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _default_sqlite_session_path() -> str:
    configured = os.environ.get("SESSION_DB_PATH")
    if configured:
        return configured
    base_dir = Path(tempfile.gettempdir()) / "travel_concierge_state"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / "adk_sessions.db")


@functools.cache
def get_session_service():
    """Creates the persistent ADK session service shared across serving routes.

    Backend selection order:
    1. Cloud SQL PostgreSQL (`INSTANCE_CONNECTION_NAME` + Secret Manager `DB_PASS`).
    2. Explicit SQLAlchemy `DATABASE_URL` (`DatabaseSessionService`).
    3. Managed `VertexAiSessionService` when `AGENT_ENGINE_ID` is configured.
    4. Persistent local `SqliteSessionService` (or `InMemorySessionService` if
       `SESSION_BACKEND=in_memory` is explicitly requested).
    """
    backend_override = os.environ.get("SESSION_BACKEND", "").strip().lower()
    if backend_override == "in_memory":
        return InMemorySessionService()

    secret_mgr = get_secret_manager()
    db_user = os.environ.get("DB_USER", "postgres")
    db_name = os.environ.get("DB_NAME", "postgres")
    db_secret_id = os.environ.get(
        "DB_PASSWORD_SECRET_ID", "travel-concierge-agent-db-password"
    )
    db_pass = secret_mgr.get_secret(
        db_secret_id, env_var_name="DB_PASS", required=False
    )
    instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME")

    if instance_connection_name and db_pass:
        encoded_user = quote(db_user, safe="")
        encoded_pass = quote(db_pass, safe="")
        encoded_instance = instance_connection_name.replace(":", "%3A")
        session_service_uri = (
            f"postgresql+asyncpg://{encoded_user}:{encoded_pass}@"
            f"/{db_name}?host=/cloudsql/{encoded_instance}"
        )
        logger.info(
            "Initialized Cloud SQL PostgreSQL persistent session service",
            extra={"event_type": "SESSION_SERVICE_INIT", "backend": "cloud_sql"},
        )
        return create_session_service_from_options(
            base_dir=_AGENT_DIR, session_service_uri=session_service_uri
        )

    if explicit_db_url := os.environ.get("DATABASE_URL"):
        logger.info(
            "Initialized SQLAlchemy DatabaseSessionService",
            extra={"event_type": "SESSION_SERVICE_INIT", "backend": "database_url"},
        )
        return create_session_service_from_options(
            base_dir=_AGENT_DIR, session_service_uri=explicit_db_url
        )

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    agent_engine_id = os.environ.get("AGENT_ENGINE_ID", "")
    if project_id and not project_id.startswith("your-") and agent_engine_id:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        logger.info(
            "Initialized VertexAiSessionService",
            extra={"event_type": "SESSION_SERVICE_INIT", "backend": "vertex_ai"},
        )
        return VertexAiSessionService(
            project=project_id,
            location=location,
            agent_engine_id=agent_engine_id,
        )

    sqlite_path = _default_sqlite_session_path()
    logger.info(
        "Initialized persistent SqliteSessionService",
        extra={
            "event_type": "SESSION_SERVICE_INIT",
            "backend": "sqlite",
            "db_path": sqlite_path,
        },
    )
    return SqliteSessionService(db_path=sqlite_path)


@functools.cache
def get_memory_service():
    """Creates the cross-session ADK Memory Bank service (Vertex AI or local)."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    agent_engine_id = os.environ.get("AGENT_ENGINE_ID", "")
    if project_id and not project_id.startswith("your-") and agent_engine_id:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        return VertexAiMemoryBankService(
            project=project_id,
            location=location,
            agent_engine_id=agent_engine_id,
        )
    return InMemoryMemoryService()


@functools.cache
def get_artifact_service():
    """Process-wide artifact service: GCS when a bucket is set, else in-memory."""
    if bucket := os.environ.get("LOGS_BUCKET_NAME"):
        return GcsArtifactService(bucket_name=bucket)
    return InMemoryArtifactService()


_registry = get_service_registry()
_registry.register_session_service("shared", lambda uri, **kw: get_session_service())
_registry.register_artifact_service("shared", lambda uri, **kw: get_artifact_service())
