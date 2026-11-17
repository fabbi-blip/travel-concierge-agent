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

"""ADK plugin capturing explicit Intent vs. Outcome structured audit telemetry.

Logs every planned agent and tool execution (`phase="intent"`) prior to
invocation and records the actual execution result, status, and latency
(`phase="outcome"`) immediately afterward, linked via `correlation_id` and
OpenTelemetry trace spans.
"""

from __future__ import annotations

import collections
import time
import uuid
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.observability.structured_logging import get_structured_logger
from app.observability.tracing import get_tracer
from app.security.pii_redactor import PiiRedactionService, get_pii_redactor

logger = get_structured_logger(__name__)


class IntentOutcomeAuditPlugin(BasePlugin):
    """Records paired pre-execution INTENT and post-execution OUTCOME records."""

    def __init__(
        self,
        *,
        name: str = "intent_outcome_audit_plugin",
        redactor: PiiRedactionService | None = None,
        max_buffered_records: int = 250,
    ) -> None:
        super().__init__(name=name)
        self._redactor = redactor or get_pii_redactor()
        self._in_flight: dict[str, tuple[str, float, Any]] = {}
        self.audit_trail: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=max_buffered_records
        )

    def _record_audit_event(self, level: str, message: str, **fields: Any) -> None:
        sanitized = self._redactor.redact_data(fields)
        self.audit_trail.append({"message": message, **sanitized})
        log_fn = getattr(logger, level.lower(), logger.info)
        log_fn(message, extra=sanitized)

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        inv_ctx = callback_context._invocation_context
        key = f"agent:{inv_ctx.invocation_id}:{agent.name}"
        correlation_id = f"corr-{uuid.uuid4().hex[:12]}"

        span = get_tracer().start_span(
            f"agent.execute.{agent.name}",
            attributes={
                "gen_ai.agent.name": agent.name,
                "gen_ai.conversation.id": inv_ctx.session.id,
                "audit.correlation_id": correlation_id,
            },
        )
        self._in_flight[key] = (correlation_id, time.perf_counter(), span)

        self._record_audit_event(
            "INFO",
            f"Agent intent declared: {agent.name}",
            event_type="AGENT_INTENT",
            phase="intent",
            correlation_id=correlation_id,
            agent_name=agent.name,
            session_id=inv_ctx.session.id,
            invocation_id=inv_ctx.invocation_id,
            intended_action=f"Invoke agent '{agent.name}'",
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        inv_ctx = callback_context._invocation_context
        key = f"agent:{inv_ctx.invocation_id}:{agent.name}"
        correlation_id, started_at, span = self._in_flight.pop(
            key, (f"corr-{uuid.uuid4().hex[:12]}", time.perf_counter(), None)
        )
        duration_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        if span is not None:
            span.set_attribute("audit.outcome_status", "completed")
            span.set_attribute("audit.duration_ms", duration_ms)
            span.end()

        self._record_audit_event(
            "INFO",
            f"Agent outcome recorded: {agent.name}",
            event_type="AGENT_OUTCOME",
            phase="outcome",
            correlation_id=correlation_id,
            agent_name=agent.name,
            session_id=inv_ctx.session.id,
            invocation_id=inv_ctx.invocation_id,
            outcome_status="completed",
            duration_ms=duration_ms,
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        call_id = tool_context.function_call_id or uuid.uuid4().hex[:10]
        key = f"tool:{call_id}:{tool.name}"
        correlation_id = f"corr-{uuid.uuid4().hex[:12]}"
        sanitized_args = self._redactor.redact_data(tool_args)

        span = get_tracer().start_span(
            f"tool.execute.{tool.name}",
            attributes={
                "gen_ai.tool.name": tool.name,
                "gen_ai.tool.call.id": str(call_id),
                "audit.correlation_id": correlation_id,
            },
        )
        self._in_flight[key] = (correlation_id, time.perf_counter(), span)

        self._record_audit_event(
            "INFO",
            f"Tool intent declared: {tool.name}",
            event_type="TOOL_INTENT",
            phase="intent",
            correlation_id=correlation_id,
            tool_name=tool.name,
            function_call_id=call_id,
            agent_name=tool_context.agent_name,
            intended_arguments=sanitized_args,
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        call_id = tool_context.function_call_id or "unknown"
        key = f"tool:{call_id}:{tool.name}"
        correlation_id, started_at, span = self._in_flight.pop(
            key, (f"corr-{uuid.uuid4().hex[:12]}", time.perf_counter(), None)
        )
        duration_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        outcome_status = (
            str(result.get("status", "success"))
            if isinstance(result, dict)
            else "success"
        )
        if span is not None:
            span.set_attribute("audit.outcome_status", outcome_status)
            span.set_attribute("audit.duration_ms", duration_ms)
            span.end()

        self._record_audit_event(
            "INFO",
            f"Tool outcome recorded: {tool.name} -> {outcome_status}",
            event_type="TOOL_OUTCOME",
            phase="outcome",
            correlation_id=correlation_id,
            tool_name=tool.name,
            function_call_id=call_id,
            agent_name=tool_context.agent_name,
            outcome_status=outcome_status,
            sanitized_outcome=self._redactor.redact_data(result),
            duration_ms=duration_ms,
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        call_id = tool_context.function_call_id or "unknown"
        key = f"tool:{call_id}:{tool.name}"
        correlation_id, started_at, span = self._in_flight.pop(
            key, (f"corr-{uuid.uuid4().hex[:12]}", time.perf_counter(), None)
        )
        duration_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        if span is not None:
            span.record_exception(error)
            span.set_attribute("audit.outcome_status", "exception")
            span.end()

        self._record_audit_event(
            "ERROR",
            f"Tool execution raised {type(error).__name__}: {tool.name}",
            event_type="TOOL_OUTCOME",
            phase="outcome",
            correlation_id=correlation_id,
            tool_name=tool.name,
            function_call_id=call_id,
            outcome_status="exception",
            error_type=type(error).__name__,
            error_detail=str(error),
            duration_ms=duration_ms,
        )
        return None
