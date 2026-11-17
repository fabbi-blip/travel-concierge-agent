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

"""Non-blocking background memory extraction, PII scrubbing, and consolidation.

Dispatches expensive preference extraction and ADK Memory Bank synchronization
via `asyncio.create_task` in `after_agent_callback` so UI streaming latency is
never blocked by memory persistence.
"""

from __future__ import annotations

import asyncio
import datetime
import functools
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from app.observability.structured_logging import get_structured_logger
from app.observability.tracing import start_sanitized_span
from app.security.pii_redactor import PiiRedactionService, get_pii_redactor

logger = get_structured_logger(__name__)

_PREFERENCE_EXTRACTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("seat_preference", re.compile(r"\b(window|aisle|middle)\s+seat\b", re.I)),
    (
        "cabin_preference",
        re.compile(r"\b(economy|premium economy|business)\s+class\b", re.I),
    ),
    (
        "dietary_preference",
        re.compile(
            r"\b(vegetarian|vegan|gluten[- ]free|halal|kosher|dairy[- ]free)\b", re.I
        ),
    ),
    (
        "flight_timing_preference",
        re.compile(r"\b(no red[- ]?eye|avoid red[- ]?eye|morning flight)\b", re.I),
    ),
    (
        "home_airport",
        re.compile(
            r"\b(?:based in|departing from|flying from|home airport(?: is)?)\s+([A-Z]{3})\b"
        ),
    ),
)


def _default_memory_db_path() -> str:
    configured = os.environ.get("TRAVEL_MEMORY_DB_PATH")
    if configured:
        return configured
    base_dir = Path(tempfile.gettempdir()) / "travel_concierge_state"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / "traveler_memory_bank.db")


class AsyncMemoryConsolidator:
    """Persists PII-redacted cross-session traveler preferences asynchronously."""

    def __init__(
        self,
        *,
        db_path: str | None = None,
        redactor: PiiRedactionService | None = None,
    ) -> None:
        self._db_path = db_path or _default_memory_db_path()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._redactor = redactor or get_pii_redactor()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traveler_memories (
                    user_id TEXT PRIMARY KEY,
                    preferences_json TEXT NOT NULL,
                    episodic_notes_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def load_traveler_profile(self, user_id: str) -> dict[str, Any]:
        """Synchronously loads consolidated preferences for prompt injection."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT preferences_json, episodic_notes_json FROM traveler_memories WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return {"preferences": {}, "recent_notes": []}
        return {
            "preferences": json.loads(row[0]),
            "recent_notes": json.loads(row[1]),
        }

    def format_profile_summary(self, user_id: str) -> str:
        """Formats a compact bullet summary of remembered preferences for prompts."""
        profile = self.load_traveler_profile(user_id)
        prefs = profile["preferences"]
        notes = profile["recent_notes"]
        if not prefs and not notes:
            return "No prior cross-session preferences recorded yet."

        parts: list[str] = []
        for key, val in sorted(prefs.items()):
            parts.append(f"- {key}: {val}")
        for note in notes[-3:]:
            parts.append(f"- prior_context: {note}")
        return "\n".join(parts)

    async def consolidate_turn_async(
        self,
        *,
        user_id: str,
        session_id: str,
        utterances: list[str],
        adk_memory_sync_coro: Any | None = None,
    ) -> dict[str, Any]:
        """Extracts preferences, scrubs PII, and writes to persistent memory off-thread.

        Args:
            user_id: Unique traveler identifier.
            session_id: Active conversation session identifier.
            utterances: User/model text turns from the completed invocation.
            adk_memory_sync_coro: Optional coroutine (`ctx.add_session_to_memory()`)
                syncing sanitized session history with ADK Memory Bank.

        Returns:
            Sanitized dictionary of consolidated preferences for `user_id`.
        """
        with start_sanitized_span(
            "memory.async_consolidate",
            attributes={"user.id": user_id, "session.id": session_id},
        ):
            sanitized_utterances = [
                self._redactor.redact_text(text) for text in utterances if text
            ]

            extracted: dict[str, str] = {}
            for utterance in sanitized_utterances:
                for pref_key, pattern in _PREFERENCE_EXTRACTORS:
                    match = pattern.search(utterance)
                    if match:
                        extracted[pref_key] = self._redactor.redact_text(
                            match.group(1).strip()
                        )

            await asyncio.to_thread(
                self._merge_and_persist_sync,
                user_id=user_id,
                extracted_preferences=extracted,
                sanitized_utterances=sanitized_utterances,
            )

            if adk_memory_sync_coro is not None:
                try:
                    await adk_memory_sync_coro
                except ValueError:
                    # Raised by ADK only if Runner has no memory_service configured.
                    pass
                except Exception as exc:
                    logger.warning(
                        "ADK Memory Bank background sync skipped",
                        extra={
                            "event_type": "MEMORY_BANK_SYNC_WARNING",
                            "session_id": session_id,
                            "error_type": type(exc).__name__,
                        },
                    )

            logger.info(
                "Completed async traveler memory consolidation",
                extra={
                    "event_type": "ASYNC_MEMORY_CONSOLIDATED",
                    "session_id": session_id,
                    "extracted_keys": sorted(extracted.keys()),
                },
            )
            return self.load_traveler_profile(user_id)

    def _merge_and_persist_sync(
        self,
        *,
        user_id: str,
        extracted_preferences: dict[str, str],
        sanitized_utterances: list[str],
    ) -> None:
        current = self.load_traveler_profile(user_id)
        merged_prefs: dict[str, Any] = dict(current["preferences"])
        merged_prefs.update(self._redactor.redact_data(extracted_preferences))

        recent_notes: list[str] = list(current["recent_notes"])
        if sanitized_utterances:
            compact_turn = self._redactor.redact_text(sanitized_utterances[-1][:180])
            if compact_turn and compact_turn not in recent_notes:
                recent_notes.append(compact_turn)
                recent_notes = recent_notes[-5:]

        now_iso = datetime.datetime.now(tz=datetime.UTC).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traveler_memories (
                    user_id, preferences_json, episodic_notes_json, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    json.dumps(merged_prefs, ensure_ascii=False),
                    json.dumps(recent_notes, ensure_ascii=False),
                    now_iso,
                ),
            )
            conn.commit()

    def schedule_consolidation(
        self,
        *,
        user_id: str,
        session_id: str,
        utterances: list[str],
        adk_memory_sync_coro: Any | None = None,
    ) -> asyncio.Task[dict[str, Any]]:
        """Schedules memory consolidation as a non-blocking `asyncio.Task`."""
        task = asyncio.create_task(
            self.consolidate_turn_async(
                user_id=user_id,
                session_id=session_id,
                utterances=utterances,
                adk_memory_sync_coro=adk_memory_sync_coro,
            ),
            name=f"consolidate_memory_{session_id}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def flush_pending_tasks(self) -> None:
        """Awaits all in-flight background memory tasks during shutdown or tests."""
        if not self._background_tasks:
            return
        pending = tuple(self._background_tasks)
        await asyncio.gather(*pending, return_exceptions=True)


@functools.lru_cache(maxsize=1)
def get_memory_consolidator() -> AsyncMemoryConsolidator:
    """Returns the singleton `AsyncMemoryConsolidator`."""
    return AsyncMemoryConsolidator()


async def preload_traveler_memory_callback(
    callback_context: CallbackContext,
) -> types.Content | None:
    """`before_agent_callback` injecting cross-session preferences into state."""
    inv_ctx = callback_context._invocation_context
    user_id = inv_ctx.user_id
    summary = get_memory_consolidator().format_profile_summary(user_id)
    callback_context.state["traveler_profile_summary"] = summary
    return None


async def schedule_background_memory_consolidation(
    callback_context: CallbackContext,
) -> types.Content | None:
    """`after_agent_callback` firing non-blocking memory consolidation."""
    inv_ctx = callback_context._invocation_context
    user_id = inv_ctx.user_id
    session = inv_ctx.session

    user_utterances: list[str] = []
    for event in session.events[-6:]:
        if event.author == "user" and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    user_utterances.append(part.text)

    sync_coro = None
    if inv_ctx.memory_service is not None:
        sync_coro = callback_context.add_session_to_memory()

    get_memory_consolidator().schedule_consolidation(
        user_id=user_id,
        session_id=session.id,
        utterances=user_utterances,
        adk_memory_sync_coro=sync_coro,
    )
    return None
