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

"""Persistent knowledge retrieval and non-blocking cross-session memory."""

from app.memory.async_memory import (
    AsyncMemoryConsolidator,
    get_memory_consolidator,
    preload_traveler_memory_callback,
    schedule_background_memory_consolidation,
)
from app.memory.knowledge_store import (
    PersistentTravelVectorStore,
    build_optional_vertex_search_tool,
    get_travel_vector_store,
)

__all__ = [
    "AsyncMemoryConsolidator",
    "PersistentTravelVectorStore",
    "build_optional_vertex_search_tool",
    "get_memory_consolidator",
    "get_travel_vector_store",
    "preload_traveler_memory_callback",
    "schedule_background_memory_consolidation",
]
