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

"""Strategic model routing across Gemini Flash (fast) and Pro (deep reasoning).

Assigns each agent role and dynamic task category to the optimal Gemini model
tier based on latency sensitivity, context complexity, and reasoning depth.
"""

from __future__ import annotations

import enum
import os
import warnings
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google.adk.models import Gemini
from google.genai import types

from app.observability.structured_logging import get_structured_logger

load_dotenv()
logger = get_structured_logger(__name__)

DEFAULT_FAST_MODEL = os.environ.get("FAST_MODEL", "gemini-3.8-flash")
DEFAULT_PRO_MODEL = os.environ.get("PRO_MODEL", "gemini-3.1-pro-preview")


def build_genai_client_kwargs() -> dict[str, Any]:
    """Builds optional `google.genai.Client` kwargs for mTLS/CAA environments.

    Standard Cloud Run, Agent Engine, and Workload Identity deployments use
    ambient Application Default Credentials (`{}`). Local workstations behind
    Enterprise Certificate / Context-Aware Access policies can set
    `USE_GCLOUD_CLI_CREDENTIALS=true` to delegate token refresh to `gcloud`.
    """
    use_gcloud = os.environ.get("USE_GCLOUD_CLI_CREDENTIALS", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    if not use_gcloud:
        return {}

    from google.oauth2 import credentials as oauth2_creds

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        creds = oauth2_creds.UserAccessTokenCredentials(quota_project_id=project_id)
    return {"credentials": creds}


class TaskComplexityTier(enum.StrEnum):
    """Logical routing tiers balancing latency, cost, and reasoning depth."""

    FAST_TRIAGE_AND_LOOKUP = "fast_triage_and_lookup"
    CONTEXT_SUMMARIZATION = "context_summarization"
    DEEP_ITINERARY_PLANNING = "deep_itinerary_planning"
    POLICY_AUDIT_AND_EXECUTION = "policy_audit_and_execution"


@dataclass(frozen=True)
class ModelRouteSpec:
    """Resolved model configuration for a specific agent capability."""

    tier: TaskComplexityTier
    model_name: str
    temperature: float
    max_output_tokens: int
    rationale: str


_MULTI_CONSTRAINT_KEYWORDS: tuple[str, ...] = (
    "itinerary",
    "optimize",
    "compare",
    "multi-city",
    "budget breakdown",
    "trade-off",
    "reimbursement",
    "reserve",
    "book",
    "schedule",
)


class StrategicModelRouter:
    """Selects Gemini Flash vs. Gemini Pro configurations per workload tier."""

    def __init__(
        self,
        *,
        fast_model: str | None = None,
        pro_model: str | None = None,
    ) -> None:
        self.fast_model = fast_model or DEFAULT_FAST_MODEL
        self.pro_model = pro_model or DEFAULT_PRO_MODEL

    def resolve_route(self, tier: TaskComplexityTier) -> ModelRouteSpec:
        """Maps a `TaskComplexityTier` to its configured model and parameters."""
        if tier == TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP:
            return ModelRouteSpec(
                tier=tier,
                model_name=self.fast_model,
                temperature=0.2,
                max_output_tokens=4096,
                rationale="Low-latency Gemini Flash tier for intent routing and parallel tool lookups.",
            )
        if tier == TaskComplexityTier.CONTEXT_SUMMARIZATION:
            return ModelRouteSpec(
                tier=tier,
                model_name=self.fast_model,
                temperature=0.1,
                max_output_tokens=2048,
                rationale="High-throughput Gemini Flash tier for deterministic history compaction.",
            )
        if tier == TaskComplexityTier.DEEP_ITINERARY_PLANNING:
            return ModelRouteSpec(
                tier=tier,
                model_name=self.pro_model,
                temperature=0.3,
                max_output_tokens=8192,
                rationale="High-reasoning Gemini Pro tier for multi-constraint itinerary and budget synthesis.",
            )
        return ModelRouteSpec(
            tier=TaskComplexityTier.POLICY_AUDIT_AND_EXECUTION,
            model_name=self.pro_model,
            temperature=0.1,
            max_output_tokens=4096,
            rationale="High-precision Gemini Pro tier for financial policy auditing and HITL bookings.",
        )

    def build_gemini_llm(self, tier: TaskComplexityTier) -> Gemini:
        """Instantiates an ADK `Gemini` model with exponential retry options."""
        spec = self.resolve_route(tier)
        extra_client_kwargs = build_genai_client_kwargs()
        logger.debug(
            "Configured Gemini model route",
            extra={
                "event_type": "MODEL_ROUTE_CONFIGURED",
                "routing_tier": spec.tier.value,
                "model_name": spec.model_name,
                "rationale": spec.rationale,
            },
        )
        return Gemini(
            model=spec.model_name,
            retry_options=types.HttpRetryOptions(attempts=3),
            client_kwargs=extra_client_kwargs or None,
        )

    def build_generation_config(
        self, tier: TaskComplexityTier
    ) -> types.GenerateContentConfig:
        """Builds the `GenerateContentConfig` matching the selected tier."""
        spec = self.resolve_route(tier)
        return types.GenerateContentConfig(
            temperature=spec.temperature,
            max_output_tokens=spec.max_output_tokens,
        )

    def classify_user_prompt(self, prompt: str) -> ModelRouteSpec:
        """Classifies a raw user query into Fast vs. Pro routing tiers."""
        normalized = prompt.lower()
        keyword_hits = sum(1 for kw in _MULTI_CONSTRAINT_KEYWORDS if kw in normalized)
        if keyword_hits >= 2 or len(prompt) > 220:
            return self.resolve_route(TaskComplexityTier.DEEP_ITINERARY_PLANNING)
        return self.resolve_route(TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP)


default_model_router = StrategicModelRouter()
