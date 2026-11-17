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

"""Multi-agent workflow definitions using ADK Parallel, Sequential, and Loop patterns."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import (
    Agent,
    BaseAgent,
    LoopAgent,
    ParallelAgent,
    SequentialAgent,
)
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from app.memory.knowledge_store import build_optional_vertex_search_tool
from app.prompts import (
    ACCOMMODATION_SPECIALIST_INSTRUCTION,
    ADVISORY_SPECIALIST_INSTRUCTION,
    BOOKING_AND_EXPENSE_EXECUTOR_INSTRUCTION,
    FLIGHT_SPECIALIST_INSTRUCTION,
    ITINERARY_ARCHITECT_PRO_INSTRUCTION,
    ITINERARY_REFINER_INSTRUCTION,
    POLICY_AUDITOR_INSTRUCTION,
)
from app.routing import StrategicModelRouter, TaskComplexityTier, default_model_router
from app.schemas import ItineraryAuditVerdict
from app.tools import (
    calculate_trip_budget_breakdown,
    lookup_airport_iata_code,
    reserve_itinerary_booking_tool,
    retrieve_destination_travel_advisories,
    search_curated_hotel_inventory,
    search_verified_flight_options,
    submit_expense_reimbursement_claim_tool,
)


class AuditEscalationChecker(BaseAgent):
    """Custom `BaseAgent` stopping `quality_assurance_loop` once audit passes."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        raw_verdict: Any = ctx.session.state.get("audit_verdict")
        approved = False

        if isinstance(raw_verdict, ItineraryAuditVerdict):
            approved = raw_verdict.approved
        elif isinstance(raw_verdict, dict):
            approved = bool(raw_verdict.get("approved", False))
        elif isinstance(raw_verdict, str) and raw_verdict.strip():
            try:
                parsed = json.loads(raw_verdict)
                if isinstance(parsed, dict):
                    approved = bool(parsed.get("approved", False))
            except json.JSONDecodeError:
                approved = False

        if approved:
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            yield Event(author=self.name)


def build_itinerary_planning_pipeline(
    router: StrategicModelRouter = default_model_router,
) -> SequentialAgent:
    """Constructs the parallel-discovery -> Pro-synthesis -> self-eval pipeline."""
    fast_cfg = router.build_generation_config(TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP)
    pro_cfg = router.build_generation_config(TaskComplexityTier.DEEP_ITINERARY_PLANNING)
    audit_cfg = router.build_generation_config(
        TaskComplexityTier.POLICY_AUDIT_AND_EXECUTION
    )

    flight_specialist = Agent(
        name="flight_research_specialist",
        description="Searches verified airline routes, cabin classes, and fares.",
        model=router.build_gemini_llm(TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP),
        generate_content_config=fast_cfg,
        instruction=FLIGHT_SPECIALIST_INSTRUCTION,
        tools=[lookup_airport_iata_code, search_verified_flight_options],
        output_key="discovered_flights",
    )

    accommodation_specialist = Agent(
        name="accommodation_specialist",
        description="Searches curated hotels matching stay dates and nightly caps.",
        model=router.build_gemini_llm(TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP),
        generate_content_config=fast_cfg,
        instruction=ACCOMMODATION_SPECIALIST_INSTRUCTION,
        tools=[search_curated_hotel_inventory],
        output_key="discovered_hotels",
    )

    advisory_tools: list[Any] = [retrieve_destination_travel_advisories]
    if vertex_search := build_optional_vertex_search_tool():
        advisory_tools.append(vertex_search)

    advisory_specialist = Agent(
        name="destination_advisory_specialist",
        description="Retrieves destination safety, visa, transit, and per-diem rules.",
        model=router.build_gemini_llm(TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP),
        generate_content_config=fast_cfg,
        instruction=ADVISORY_SPECIALIST_INSTRUCTION,
        tools=advisory_tools,
        output_key="destination_advisories",
    )

    parallel_discovery_squad = ParallelAgent(
        name="parallel_discovery_squad",
        description="Executes flight, hotel, and advisory discovery concurrently.",
        sub_agents=[
            flight_specialist,
            accommodation_specialist,
            advisory_specialist,
        ],
    )

    itinerary_architect_pro = Agent(
        name="itinerary_architect_pro",
        description="Synthesizes multi-constraint travel itineraries and itemized budgets.",
        model=router.build_gemini_llm(TaskComplexityTier.DEEP_ITINERARY_PLANNING),
        generate_content_config=pro_cfg,
        instruction=ITINERARY_ARCHITECT_PRO_INSTRUCTION,
        tools=[calculate_trip_budget_breakdown],
        output_key="draft_itinerary",
    )

    policy_and_budget_auditor = Agent(
        name="policy_and_budget_auditor",
        description="Evaluates draft itineraries against budget, safety, and PII rules.",
        model=router.build_gemini_llm(TaskComplexityTier.POLICY_AUDIT_AND_EXECUTION),
        generate_content_config=audit_cfg,
        instruction=POLICY_AUDITOR_INSTRUCTION,
        output_schema=ItineraryAuditVerdict,
        output_key="audit_verdict",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    itinerary_refiner = Agent(
        name="itinerary_refiner",
        description="Resolves audit findings and produces the verified final itinerary.",
        model=router.build_gemini_llm(TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP),
        generate_content_config=fast_cfg,
        instruction=ITINERARY_REFINER_INSTRUCTION,
        tools=[calculate_trip_budget_breakdown],
        output_key="draft_itinerary",
    )

    quality_assurance_loop = LoopAgent(
        name="quality_assurance_loop",
        description="Iteratively audits and refines itinerary compliance up to 2 passes.",
        sub_agents=[
            policy_and_budget_auditor,
            itinerary_refiner,
            AuditEscalationChecker(name="audit_escalation_checker"),
        ],
        max_iterations=2,
    )

    return SequentialAgent(
        name="itinerary_planning_pipeline",
        description=(
            "End-to-end trip planning workflow: concurrent flight/hotel/advisory "
            "discovery, Gemini Pro budget synthesis, and self-evaluation QA loop."
        ),
        sub_agents=[
            parallel_discovery_squad,
            itinerary_architect_pro,
            quality_assurance_loop,
        ],
    )


async def _summarize_pending_hitl_confirmation(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Emits a clear user-facing summary whenever ADK pauses on `adk_request_confirmation`."""
    inv_ctx = callback_context._invocation_context
    for event in reversed(inv_ctx.session.events[-6:]):
        if event.invocation_id != inv_ctx.invocation_id:
            continue
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            fn_call = part.function_call
            if fn_call and fn_call.name == "adk_request_confirmation":
                orig = (fn_call.args or {}).get("originalFunctionCall") or {}
                tool_name = orig.get("name", "financial_action")
                orig_args = orig.get("args") or {}
                amount = orig_args.get("total_authorized_usd") or orig_args.get(
                    "amount_usd"
                )
                details: list[str] = []
                if fq := orig_args.get("flight_quote_id"):
                    details.append(f"flight `{fq}`")
                if hq := orig_args.get("hotel_quote_id"):
                    details.append(f"hotel `{hq}`")
                if ref := orig_args.get("trip_reference"):
                    details.append(f"trip `{ref}`")
                amount_str = (
                    f"${float(amount):,.2f} USD" if amount is not None else "the requested amount"
                )
                scope_str = f" ({', '.join(details)})" if details else ""
                summary = (
                    f"Explicit human confirmation is required before charging or "
                    f"submitting {amount_str}{scope_str} via `{tool_name}`. "
                    "Please approve or reject the confirmation prompt to proceed."
                )
                return types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=summary)],
                )
    return None


def build_booking_and_expense_executor(
    router: StrategicModelRouter = default_model_router,
) -> Agent:
    """Constructs the high-stakes HITL booking and expense compliance specialist."""
    exec_cfg = router.build_generation_config(
        TaskComplexityTier.POLICY_AUDIT_AND_EXECUTION
    )
    return Agent(
        name="booking_and_expense_executor",
        description=(
            "Executes confirmed flight/hotel reservations and post-trip expense "
            "reimbursement claims behind Human-in-the-Loop approval gates."
        ),
        model=router.build_gemini_llm(TaskComplexityTier.POLICY_AUDIT_AND_EXECUTION),
        generate_content_config=exec_cfg,
        instruction=BOOKING_AND_EXPENSE_EXECUTOR_INSTRUCTION,
        tools=[
            reserve_itinerary_booking_tool,
            submit_expense_reimbursement_claim_tool,
            calculate_trip_budget_breakdown,
        ],
        after_agent_callback=_summarize_pending_hitl_confirmation,
    )
