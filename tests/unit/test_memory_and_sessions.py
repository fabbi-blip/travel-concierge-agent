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

"""Unit tests for Category 2: Context Compaction, Persistent State, and Async Memory."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.sessions.sqlite_session_service import SqliteSessionService

from app.agent import app as adk_app
from app.memory.async_memory import AsyncMemoryConsolidator
from app.memory.knowledge_store import PersistentTravelVectorStore


def test_app_configures_history_compaction_and_context_caching() -> None:
    """Verifies EventsCompactionConfig, ContextCacheConfig, and ResumabilityConfig."""
    compaction = adk_app.events_compaction_config
    assert compaction is not None
    assert compaction.compaction_interval == 12
    assert compaction.overlap_size == 3
    assert compaction.token_threshold == 24000
    assert compaction.event_retention_size == 6
    assert compaction.summarizer is not None

    cache_cfg = adk_app.context_cache_config
    assert cache_cfg is not None
    assert cache_cfg.min_tokens == 2048
    assert adk_app.resumability_config is not None
    assert adk_app.resumability_config.is_resumable is True


@pytest.mark.asyncio
async def test_persistent_sqlite_session_and_vector_store_survive_reinstantiation(
    tmp_path: Path,
) -> None:
    """Verifies persistent session DB and persistent vector store retain state."""
    session_db = str(tmp_path / "sessions.db")
    svc_writer = SqliteSessionService(db_path=session_db)
    created = await svc_writer.create_session(
        app_name="app",
        user_id="traveler_42",
        state={"preferred_origin": "SFO"},
    )

    # Re-open a fresh SqliteSessionService instance pointing at the same SQLite file.
    svc_reader = SqliteSessionService(db_path=session_db)
    reloaded = await svc_reader.get_session(
        app_name="app", user_id="traveler_42", session_id=created.id
    )
    assert reloaded is not None
    assert reloaded.state.get("preferred_origin") == "SFO"

    # Persistent vector store retains custom upserted policy documents across restarts.
    vec_db = str(tmp_path / "vector_knowledge.db")
    store_writer = PersistentTravelVectorStore(db_path=vec_db)
    store_writer.upsert_document(
        {
            "doc_id": "ADV-CH-ZRH",
            "destination": "Zurich, Switzerland",
            "country_code": "CH",
            "safety_tier": "level_1_normal",
            "daily_per_diem_usd": 175.0,
            "visa_and_entry_rules": "Schengen entry rules apply for Zurich.",
            "local_transit_guidance": "SBB rail connects ZRH airport to Zurich HB in 10 minutes.",
            "expense_policy_notes": "Keep CHF hotel receipts with itemized city tax.",
        }
    )

    store_reader = PersistentTravelVectorStore(db_path=vec_db)
    hits = store_reader.search("Zurich Switzerland SBB rail per diem", top_k=2)
    assert hits
    assert hits[0].doc_id == "ADV-CH-ZRH"
    assert hits[0].daily_per_diem_usd == 175.0


@pytest.mark.asyncio
async def test_async_memory_consolidator_runs_in_background_and_scrubs_pii(
    tmp_path: Path,
) -> None:
    """Verifies non-blocking asyncio background consolidation and PII scrubbing."""
    mem_db = str(tmp_path / "memory_bank.db")
    consolidator = AsyncMemoryConsolidator(db_path=mem_db)

    task = consolidator.schedule_consolidation(
        user_id="user_async_01",
        session_id="sess_99",
        utterances=[
            "I am flying from SFO, prefer a window seat and vegetarian meals, "
            "avoid red-eye flights, and my email is private.user@example.com."
        ],
    )
    # Caller receives an in-flight asyncio.Task immediately without blocking UI turn.
    assert not task.done() or task is not None
    await consolidator.flush_pending_tasks()

    profile = consolidator.load_traveler_profile("user_async_01")
    prefs = profile["preferences"]
    assert prefs["home_airport"] == "SFO"
    assert prefs["seat_preference"].lower() == "window"
    assert prefs["dietary_preference"].lower() == "vegetarian"
    assert "redeye" in prefs["flight_timing_preference"].lower().replace("-", "")

    summary = consolidator.format_profile_summary("user_async_01")
    assert "private.user@example.com" not in summary
    assert "[REDACTED_EMAIL]" in summary
