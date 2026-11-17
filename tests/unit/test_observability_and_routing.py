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

"""Unit tests for Category 4 & 5: Structured Logging, Intent/Outcome, Tracing, Routing, and Golden Evals."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.adk.tools import FunctionTool

from app.observability.structured_logging import StructuredJsonFormatter
from app.observability.tracing import (
    configure_distributed_tracing,
    get_in_memory_span_exporter,
    start_sanitized_span,
)
from app.plugins.intent_outcome_logger import IntentOutcomeAuditPlugin
from app.routing import StrategicModelRouter, TaskComplexityTier
from app.tools import lookup_airport_iata_code
from tests.eval.golden_regression_harness import run_golden_regression_suite


def test_structured_json_formatter_and_zero_print_statements_in_app() -> None:
    """Verifies JSON log structure, PII redaction, and absence of bare print() in app/."""
    formatter = StructuredJsonFormatter(
        project_id="demo-obs-project", service_name="travel-concierge-agent"
    )

    configure_distributed_tracing(enable_in_memory_exporter=True)
    with start_sanitized_span("unit.log_correlation_test"):
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Processing receipt for card 4111-2222-3333-4444",
            args=(),
            exc_info=None,
        )
        record.event_type = "UNIT_TEST_EVENT"
        record.traveler_email = "alice@example.com"
        serialized = formatter.format(record)

    parsed = json.loads(serialized)
    assert parsed["severity"] == "INFO"
    assert parsed["service"] == "travel-concierge-agent"
    assert parsed["event_type"] == "UNIT_TEST_EVENT"
    assert "[REDACTED_PAYMENT_CARD]" in parsed["message"]
    assert parsed["traveler_email"] == "[REDACTED_EMAIL]"
    assert "logging.googleapis.com/trace" in parsed
    assert "logging.googleapis.com/spanId" in parsed

    # Statically verify zero bare `print(...)` calls exist in production `app/` modules.
    app_dir = Path(__file__).resolve().parents[2] / "app"
    for py_file in app_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                pytest.fail(f"Unstructured print() call found in {py_file}:{node.lineno}")


@pytest.mark.asyncio
async def test_intent_vs_outcome_plugin_captures_paired_correlated_records() -> None:
    """Verifies IntentOutcomeAuditPlugin logs pre-execution intent and post-execution outcome."""
    plugin = IntentOutcomeAuditPlugin()
    tool = FunctionTool(lookup_airport_iata_code)
    tool_ctx = SimpleNamespace(
        function_call_id="call_abc123",
        agent_name="travel_concierge_agent",
    )

    await plugin.before_tool_callback(
        tool=tool,
        tool_args={"city_or_airport_query": "Tokyo (contact user@example.com)"},
        tool_context=tool_ctx,  # type: ignore[arg-type]
    )
    await plugin.after_tool_callback(
        tool=tool,
        tool_args={"city_or_airport_query": "Tokyo"},
        tool_context=tool_ctx,  # type: ignore[arg-type]
        result={"status": "success", "matches": [{"iata_code": "NRT"}]},
    )

    assert len(plugin.audit_trail) == 2
    intent_event, outcome_event = list(plugin.audit_trail)
    assert intent_event["phase"] == "intent"
    assert intent_event["event_type"] == "TOOL_INTENT"
    assert (
        intent_event["intended_arguments"]["city_or_airport_query"]
        == "Tokyo (contact [REDACTED_EMAIL])"
    )

    assert outcome_event["phase"] == "outcome"
    assert outcome_event["event_type"] == "TOOL_OUTCOME"
    assert outcome_event["outcome_status"] == "success"
    assert outcome_event["correlation_id"] == intent_event["correlation_id"]
    assert outcome_event["duration_ms"] >= 0.0


def test_distributed_tracing_links_parent_and_child_spans() -> None:
    """Verifies OpenTelemetry parent-child span linkage and PII scrubbing on attributes."""
    configure_distributed_tracing(enable_in_memory_exporter=True)
    exporter = get_in_memory_span_exporter()
    assert exporter is not None
    exporter.clear()

    with start_sanitized_span(
        "concierge.request", attributes={"user.query": "Email me at bob@example.com"}
    ):
        lookup_airport_iata_code("Tokyo")

    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert "concierge.request" in by_name
    assert "tool.lookup_airport_iata_code" in by_name

    parent_span = by_name["concierge.request"]
    child_span = by_name["tool.lookup_airport_iata_code"]
    assert child_span.parent is not None
    assert child_span.parent.span_id == parent_span.context.span_id
    assert parent_span.attributes["user.query"] == "Email me at [REDACTED_EMAIL]"


def test_strategic_model_routing_assigns_flash_and_pro_appropriately() -> None:
    """Verifies fast tasks route to Gemini Flash and complex planning routes to Gemini Pro."""
    router = StrategicModelRouter(
        fast_model="gemini-2.5-flash", pro_model="gemini-2.5-pro"
    )

    fast_spec = router.classify_user_prompt("What is the IATA code for London?")
    assert fast_spec.model_name == "gemini-2.5-flash"
    assert fast_spec.tier == TaskComplexityTier.FAST_TRIAGE_AND_LOOKUP

    pro_spec = router.classify_user_prompt(
        "Build and optimize a multi-city itinerary and budget breakdown comparing "
        "Tokyo and Singapore hotels with schedule constraints."
    )
    assert pro_spec.model_name == "gemini-2.5-pro"
    assert pro_spec.tier == TaskComplexityTier.DEEP_ITINERARY_PLANNING


@pytest.mark.asyncio
async def test_golden_evaluation_regression_suite_passes_all_cases() -> None:
    """Runs the static golden evaluation harness end-to-end through ADK Runner."""
    results = await run_golden_regression_suite()
    assert len(results) == 4
    failures = [r for r in results if not r.passed]
    assert not failures, f"Golden regression failures: {failures}"
