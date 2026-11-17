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

"""Persistent vector store and Vertex AI Search integration for travel policies.

Stores destination advisories, visa requirements, and per-diem expense policies
in a SQLite database indexed with TF-IDF cosine-similarity vectors, and exposes
native `VertexAiSearchTool` grounding when a Google Cloud datastore is configured.
"""

from __future__ import annotations

import functools
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from google.adk.tools import VertexAiSearchTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.observability.structured_logging import get_structured_logger
from app.observability.tracing import traced_span
from app.schemas import AdvisoryDocumentHit

logger = get_structured_logger(__name__)

_SEED_ADVISORY_DOCUMENTS: tuple[dict[str, Any], ...] = (
    {
        "doc_id": "ADV-JP-TYO",
        "destination": "Tokyo, Japan",
        "country_code": "JP",
        "safety_tier": "level_1_normal",
        "daily_per_diem_usd": 145.0,
        "visa_and_entry_rules": (
            "Visa-exempt up to 90 days for US/EU/UK tourist and short business "
            "passports; complete Visit Japan Web QR registration prior to arrival."
        ),
        "local_transit_guidance": (
            "Use Narita Express (N'EX) or Keisei Skyliner from NRT and Tokyo Monorail "
            "or Keikyu Line from HND. Load mobile Suica/Pasmo IC card for subways."
        ),
        "expense_policy_notes": (
            "Itemized hotel folios and Shinkansen/rail receipts over $25 USD are "
            "required. Standard lodging guideline in central Tokyo is <= $380/night."
        ),
    },
    {
        "doc_id": "ADV-GB-LON",
        "destination": "London, United Kingdom",
        "country_code": "GB",
        "safety_tier": "level_1_normal",
        "daily_per_diem_usd": 160.0,
        "visa_and_entry_rules": (
            "Electronic Travel Authorisation (UK ETA) required prior to boarding for "
            "eligible visa-exempt nationals; passport must be valid for entire stay."
        ),
        "local_transit_guidance": (
            "Elizabeth Line and Heathrow Express connect LHR to Paddington and the "
            "City. Contactless tap-to-pay enforces automatic TfL daily fare caps."
        ),
        "expense_policy_notes": (
            "VAT invoices should be retained for hotel and conference charges. "
            "Standard central London lodging cap is <= $410/night."
        ),
    },
    {
        "doc_id": "ADV-US-NYC",
        "destination": "New York City, United States",
        "country_code": "US",
        "safety_tier": "level_1_normal",
        "daily_per_diem_usd": 135.0,
        "visa_and_entry_rules": (
            "REAL ID-compliant driver's license or passport required for domestic TSA "
            "screening; ESTA/B1-B2 required for international visitors."
        ),
        "local_transit_guidance": (
            "AirTrain JFK + LIRR to Penn Station or Grand Central Madison avoids peak "
            "tunnel congestion; OMNY contactless tap works on all MTA subways."
        ),
        "expense_policy_notes": (
            "Rideshare surge fares above $95 between Manhattan and JFK/EWR require "
            "brief justification note; transit rail is preferred."
        ),
    },
    {
        "doc_id": "ADV-FR-PAR",
        "destination": "Paris, France",
        "country_code": "FR",
        "safety_tier": "level_2_caution",
        "daily_per_diem_usd": 150.0,
        "visa_and_entry_rules": (
            "Schengen 90/180-day rule applies; passport must have >= 3 months validity "
            "beyond planned Schengen departure date."
        ),
        "local_transit_guidance": (
            "RER B links CDG to central Paris; exercise heightened awareness against "
            "pickpockets on crowded metro lines and major rail terminals."
        ),
        "expense_policy_notes": (
            "City tourist tax (taxe de sejour) is reimbursable as part of lodging "
            "when itemized on the hotel invoice."
        ),
    },
    {
        "doc_id": "ADV-SG-SIN",
        "destination": "Singapore",
        "country_code": "SG",
        "safety_tier": "level_1_normal",
        "daily_per_diem_usd": 155.0,
        "visa_and_entry_rules": (
            "Submit electronic SG Arrival Card (SGAC) within 3 days before arrival; "
            "passport requires >= 6 months remaining validity."
        ),
        "local_transit_guidance": (
            "MRT East-West/Downtown lines and Grab rideshare connect Changi Airport "
            "(SIN) to Marina Bay and CBD in 20-35 minutes."
        ),
        "expense_policy_notes": (
            "Strict local laws apply to prohibited imports; keep GST-inclusive "
            "receipts for all ground transport and client meals."
        ),
    },
)


def _default_knowledge_db_path() -> str:
    configured = os.environ.get("TRAVEL_KNOWLEDGE_DB_PATH")
    if configured:
        return configured
    base_dir = Path(tempfile.gettempdir()) / "travel_concierge_state"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / "travel_knowledge_vector.db")


class PersistentTravelVectorStore:
    """SQLite-backed document store with TF-IDF cosine-similarity vector search."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _default_knowledge_db_path()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._records: list[dict[str, Any]] = []
        self._matrix: Any = None
        self._initialize_database()
        self._rebuild_vector_index()

    def _initialize_database(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS travel_advisory_docs (
                    doc_id TEXT PRIMARY KEY,
                    destination TEXT NOT NULL,
                    country_code TEXT NOT NULL,
                    safety_tier TEXT NOT NULL,
                    daily_per_diem_usd REAL NOT NULL,
                    visa_and_entry_rules TEXT NOT NULL,
                    local_transit_guidance TEXT NOT NULL,
                    expense_policy_notes TEXT NOT NULL
                )
                """
            )
            cursor = conn.execute("SELECT COUNT(*) FROM travel_advisory_docs")
            count = int(cursor.fetchone()[0])
            if count == 0:
                conn.executemany(
                    """
                    INSERT INTO travel_advisory_docs (
                        doc_id, destination, country_code, safety_tier,
                        daily_per_diem_usd, visa_and_entry_rules,
                        local_transit_guidance, expense_policy_notes
                    ) VALUES (
                        :doc_id, :destination, :country_code, :safety_tier,
                        :daily_per_diem_usd, :visa_and_entry_rules,
                        :local_transit_guidance, :expense_policy_notes
                    )
                    """,
                    _SEED_ADVISORY_DOCUMENTS,
                )
                conn.commit()

    def _rebuild_vector_index(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM travel_advisory_docs ORDER BY doc_id"
            ).fetchall()

        self._records = [dict(row) for row in rows]
        corpus = [
            " ".join(
                [
                    rec["destination"],
                    rec["country_code"],
                    rec["safety_tier"],
                    rec["visa_and_entry_rules"],
                    rec["local_transit_guidance"],
                    rec["expense_policy_notes"],
                ]
            )
            for rec in self._records
        ]
        if corpus:
            self._matrix = self._vectorizer.fit_transform(corpus)

    def upsert_document(self, doc: dict[str, Any]) -> None:
        """Inserts or updates an advisory document and refreshes the vector index."""
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO travel_advisory_docs (
                        doc_id, destination, country_code, safety_tier,
                        daily_per_diem_usd, visa_and_entry_rules,
                        local_transit_guidance, expense_policy_notes
                    ) VALUES (
                        :doc_id, :destination, :country_code, :safety_tier,
                        :daily_per_diem_usd, :visa_and_entry_rules,
                        :local_transit_guidance, :expense_policy_notes
                    )
                    """,
                    doc,
                )
                conn.commit()
            self._rebuild_vector_index()

    @traced_span("vector_store.semantic_search")
    def search(self, query: str, *, top_k: int = 3) -> list[AdvisoryDocumentHit]:
        """Executes cosine-similarity vector retrieval against persisted advisories."""
        with self._lock:
            if not self._records or self._matrix is None:
                return []
            query_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._matrix).flatten()
            ranked_indices = np.argsort(scores)[::-1][:top_k]

            hits: list[AdvisoryDocumentHit] = []
            for idx in ranked_indices:
                score = float(scores[int(idx)])
                if score <= 0.0:
                    continue
                rec = self._records[int(idx)]
                hits.append(
                    AdvisoryDocumentHit(
                        doc_id=rec["doc_id"],
                        destination=rec["destination"],
                        country_code=rec["country_code"],
                        safety_tier=rec["safety_tier"],
                        daily_per_diem_usd=float(rec["daily_per_diem_usd"]),
                        visa_and_entry_rules=rec["visa_and_entry_rules"],
                        local_transit_guidance=rec["local_transit_guidance"],
                        expense_policy_notes=rec["expense_policy_notes"],
                        relevance_score=round(min(1.0, score), 4),
                    )
                )
            return hits

    def list_available_destinations(self) -> list[str]:
        """Returns all indexed destination names for guided error recovery."""
        return [rec["destination"] for rec in self._records]


@functools.lru_cache(maxsize=1)
def get_travel_vector_store() -> PersistentTravelVectorStore:
    """Returns the singleton persistent travel advisory vector store."""
    return PersistentTravelVectorStore()


def build_optional_vertex_search_tool() -> VertexAiSearchTool | None:
    """Creates a `VertexAiSearchTool` when `VERTEX_SEARCH_DATASTORE_ID` is set.

    Enables enterprise grounding against Vertex AI Search (Discovery Engine) in
    deployed Google Cloud environments while keeping local execution dependency-free.
    """
    datastore_id = os.environ.get("VERTEX_SEARCH_DATASTORE_ID", "").strip()
    if not datastore_id or datastore_id.startswith("your-"):
        return None
    logger.info(
        "Enabling VertexAiSearchTool grounding",
        extra={
            "event_type": "VERTEX_SEARCH_ENABLED",
            "datastore_id": datastore_id,
        },
    )
    return VertexAiSearchTool(
        data_store_id=datastore_id,
        bypass_multi_tools_limit=True,
    )
