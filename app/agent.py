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

"""Root multi-agent coordinator and ADK `App` configuration.

Wires strategic Gemini Flash/Pro routing, context compaction, context caching,
resumable Human-in-the-Loop execution, security guardrails, structured
Intent/Outcome audit logging, and non-blocking cross-session memory consolidation.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import App, EventsCompactionConfig, ResumabilityConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.cloud import bigquery

from app.memory import (
    preload_traveler_memory_callback,
    schedule_background_memory_consolidation,
)
from app.observability import (
    configure_distributed_tracing,
    configure_structured_logging,
    get_structured_logger,
)
from app.plugins import IntentOutcomeAuditPlugin, TravelPolicyGuardrailPlugin
from app.prompts import COORDINATOR_CONSTITUTION
from app.routing import StrategicModelRouter, TaskComplexityTier, default_model_router
from app.sub_agents import (
    build_booking_and_expense_executor,
    build_itinerary_planning_pipeline,
)
from app.tools import (
    lookup_airport_iata_code,
    retrieve_destination_travel_advisories,
)

configure_structured_logging()
configure_distributed_tracing()
logger = get_structured_logger(__name__)


def create_root_agent(
    router: StrategicModelRouter = default_model_router,
) -> Agent:
    """Creates the root `travel_concierge_agent` coordinator hierarchy."""
    coordinator_cfg = router.build_generation_config(
        TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP
    )
    return Agent(
        # Must remain synced with `create_params.root_agent_name` in agents-cli-manifest.yaml.
        name="travel_concierge_agent",
        description=(
            "Smart Travel & Expense Concierge coordinating concurrent itinerary "
            "research, Gemini Pro budget synthesis, and HITL booking/expense filing."
        ),
        model=router.build_gemini_llm(TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP),
        generate_content_config=coordinator_cfg,
        instruction=COORDINATOR_CONSTITUTION,
        tools=[
            PreloadMemoryTool(),
            lookup_airport_iata_code,
            retrieve_destination_travel_advisories,
        ],
        sub_agents=[
            build_itinerary_planning_pipeline(router=router),
            build_booking_and_expense_executor(router=router),
        ],
        before_agent_callback=preload_traveler_memory_callback,
        after_agent_callback=schedule_background_memory_consolidation,
    )


def _build_application_plugins() -> list[BasePlugin]:
    """Assembles global security, telemetry, retry, and context-filter plugins."""
    plugins: list[BasePlugin] = [
        TravelPolicyGuardrailPlugin(),
        IntentOutcomeAuditPlugin(),
        ReflectAndRetryToolPlugin(
            max_retries=2,
            throw_exception_if_retry_exceeded=False,
        ),
        ContextFilterPlugin(num_invocations_to_keep=20),
    ]

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    enable_bq = os.environ.get("ENABLE_BQ_ANALYTICS", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    if project_id and not project_id.startswith("your-") and enable_bq:
        dataset_id = os.environ.get("BQ_ANALYTICS_DATASET_ID", "adk_agent_analytics")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        try:
            bq = bigquery.Client(project=project_id)
            bq.create_dataset(f"{project_id}.{dataset_id}", exists_ok=True)
            plugins.append(
                BigQueryAgentAnalyticsPlugin(
                    project_id=project_id,
                    dataset_id=dataset_id,
                    location=location,
                    config=BigQueryLoggerConfig(
                        gcs_bucket_name=os.environ.get("BQ_ANALYTICS_GCS_BUCKET"),
                        connection_id=os.environ.get("BQ_ANALYTICS_CONNECTION_ID"),
                    ),
                )
            )
        except Exception as exc:
            logger.warning(
                "BigQuery Agent Analytics plugin skipped",
                extra={
                    "event_type": "BQ_ANALYTICS_INIT_SKIPPED",
                    "error_type": type(exc).__name__,
                },
            )

    return plugins


root_agent = create_root_agent()

app = App(
    name="app",
    root_agent=root_agent,
    plugins=_build_application_plugins(),
    events_compaction_config=EventsCompactionConfig(
        summarizer=LlmEventSummarizer(
            llm=default_model_router.build_gemini_llm(
                TaskComplexityTier.CONTEXT_SUMMARIZATION
            )
        ),
        compaction_interval=12,
        overlap_size=3,
        token_threshold=24000,
        event_retention_size=6,
    ),
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=1800,
        cache_intervals=10,
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)
