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

"""Security, policy compliance, and output self-evaluation ADK guardrail plugin.

Enforces three defense-in-depth layers across all agents in the `App`:
1. Input Guardrail (`before_model_callback`): Screens prompts for injection,
   credential exfiltration, and prohibited fraud requests (with optional Google
   Cloud Model Armor template screening).
2. Tool Policy Guardrail (`before_tool_callback`): Enforces hard financial
   ceilings on reservations and expense claims with guided recovery feedback.
3. Output Self-Evaluation Guardrail (`after_model_callback`): Scrubs any leaked
   PII from generated LLM responses prior to user delivery.
"""

from __future__ import annotations

import os
import re
from typing import Any, Protocol

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.observability.structured_logging import get_structured_logger
from app.security.pii_redactor import PiiRedactionService, get_pii_redactor

logger = get_structured_logger(__name__)

_PROMPT_INJECTION_AND_ABUSE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "PROMPT_INJECTION_ATTEMPT",
        re.compile(
            r"(?i)\b(?:ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions|"
            r"reveal\s+(?:your\s+)?system\s+prompt|"
            r"bypass\s+(?:human\s+)?confirmation|"
            r"dump\s+(?:all\s+)?(?:environment\s+variables|secrets?))\b"
        ),
    ),
    (
        "PROHIBITED_FRAUDULENT_TRAVEL_REQUEST",
        re.compile(
            r"(?i)\b(?:forge\s+(?:a\s+)?receipt|fake\s+(?:hotel\s+)?invoice|"
            r"counterfeit\s+passport|evade\s+customs\s+sanctions)\b"
        ),
    ),
)


class ModelArmorClientProtocol(Protocol):
    """Protocol for optional Google Cloud Model Armor template screening."""

    def sanitize_user_prompt(self, request: Any) -> Any:
        """Evaluates user text against a Google Cloud Model Armor template."""


class TravelPolicyGuardrailPlugin(BasePlugin):
    """Application-level security, financial policy, and PII output guardrail."""

    def __init__(
        self,
        *,
        name: str = "travel_policy_guardrail_plugin",
        max_single_booking_usd: float = 15000.0,
        max_single_expense_claim_usd: float = 5000.0,
        model_armor_template_id: str | None = None,
        model_armor_client: ModelArmorClientProtocol | None = None,
        redactor: PiiRedactionService | None = None,
    ) -> None:
        super().__init__(name=name)
        self.max_single_booking_usd = max_single_booking_usd
        self.max_single_expense_claim_usd = max_single_expense_claim_usd
        self._model_armor_template_id = (
            model_armor_template_id
            if model_armor_template_id is not None
            else os.environ.get("MODEL_ARMOR_TEMPLATE_ID", "").strip()
        )
        self._model_armor_client = model_armor_client
        self._redactor = redactor or get_pii_redactor()

    def _resolve_model_armor_client(self) -> ModelArmorClientProtocol | None:
        if self._model_armor_client is not None:
            return self._model_armor_client
        if not self._model_armor_template_id:
            return None
        try:
            from google.cloud import modelarmor_v1

            self._model_armor_client = modelarmor_v1.ModelArmorClient()
            return self._model_armor_client
        except Exception:
            return None

    def _extract_latest_user_text(self, llm_request: LlmRequest) -> str:
        if not llm_request.contents:
            return ""
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts:
                texts = [part.text for part in content.parts if part.text]
                if texts:
                    return "\n".join(texts)
        return ""

    def _check_model_armor(self, user_text: str) -> str | None:
        client = self._resolve_model_armor_client()
        if client is None or not self._model_armor_template_id or not user_text:
            return None
        try:
            response = client.sanitize_user_prompt(
                request={
                    "name": self._model_armor_template_id,
                    "user_prompt_data": {"text": user_text},
                }
            )
            result = getattr(response, "sanitization_result", None)
            match_state = getattr(result, "filter_match_state", None)
            if match_state is not None and "MATCH_FOUND" in str(match_state):
                return "MODEL_ARMOR_POLICY_BLOCK"
        except Exception as exc:
            logger.warning(
                "Model Armor API check failed; deterministic guardrails remain active",
                extra={
                    "event_type": "MODEL_ARMOR_WARNING",
                    "error_type": type(exc).__name__,
                },
            )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        user_text = self._extract_latest_user_text(llm_request)
        if not user_text:
            return None

        violation_code: str | None = None
        for code, pattern in _PROMPT_INJECTION_AND_ABUSE_PATTERNS:
            if pattern.search(user_text):
                violation_code = code
                break

        if violation_code is None:
            violation_code = self._check_model_armor(user_text)

        if violation_code is not None:
            logger.warning(
                "Blocked request via input security guardrail",
                extra={
                    "event_type": "GUARDRAIL_BLOCKED_INPUT",
                    "violation_code": violation_code,
                    "agent_name": callback_context.agent_name,
                },
            )
            refusal = (
                f"Request blocked by security policy ({violation_code}). "
                "I can only assist with legitimate travel planning, verified "
                "bookings, and compliant expense reimbursements."
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=refusal)],
                )
            )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        if tool.name == "reserve_itinerary_booking":
            amount = float(tool_args.get("total_authorized_usd", 0.0) or 0.0)
            if amount > self.max_single_booking_usd:
                logger.warning(
                    "Blocked over-limit reservation in tool policy guardrail",
                    extra={
                        "event_type": "GUARDRAIL_BLOCKED_TOOL",
                        "tool_name": tool.name,
                        "requested_usd": amount,
                        "ceiling_usd": self.max_single_booking_usd,
                    },
                )
                return {
                    "status": "error",
                    "error_code": "EXECUTIVE_BOOKING_CAP_EXCEEDED",
                    "message": (
                        f"Requested reservation amount (${amount:,.2f} USD) exceeds the "
                        f"automated concierge limit of ${self.max_single_booking_usd:,.2f} USD."
                    ),
                    "recoverable": True,
                    "recovery_instructions": (
                        "Select lower-cost economy/premium-economy flight or hotel quotes "
                        f"so `total_authorized_usd` is <= ${self.max_single_booking_usd:,.2f}, "
                        "or split multi-trip reservations."
                    ),
                }

        if tool.name == "submit_expense_reimbursement_claim":
            claim_amount = float(tool_args.get("amount_usd", 0.0) or 0.0)
            if claim_amount > self.max_single_expense_claim_usd:
                logger.warning(
                    "Blocked over-limit expense claim in tool policy guardrail",
                    extra={
                        "event_type": "GUARDRAIL_BLOCKED_TOOL",
                        "tool_name": tool.name,
                        "requested_usd": claim_amount,
                        "ceiling_usd": self.max_single_expense_claim_usd,
                    },
                )
                return {
                    "status": "error",
                    "error_code": "SINGLE_CLAIM_CAP_EXCEEDED",
                    "message": (
                        f"Expense claim of ${claim_amount:,.2f} USD exceeds the automated "
                        f"reimbursement threshold of ${self.max_single_expense_claim_usd:,.2f} USD."
                    ),
                    "recoverable": True,
                    "recovery_instructions": (
                        "Itemize individual lodging, airfare, and meal receipts separately "
                        f"so each claim is <= ${self.max_single_expense_claim_usd:,.2f} USD."
                    ),
                }

        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if not llm_response.content or not llm_response.content.parts:
            return None

        modified = False
        new_parts: list[types.Part] = []
        for part in llm_response.content.parts:
            if part.text and self._redactor.contains_pii(part.text):
                scrubbed_text = self._redactor.redact_text(part.text)
                new_parts.append(types.Part.from_text(text=scrubbed_text))
                modified = True
            else:
                new_parts.append(part)

        if not modified:
            return None

        logger.info(
            "Scrubbed sensitive PII from model output via self-eval guardrail",
            extra={
                "event_type": "GUARDRAIL_REDACTED_MODEL_OUTPUT",
                "agent_name": callback_context.agent_name,
            },
        )
        return LlmResponse(
            content=types.Content(
                role=llm_response.content.role or "model",
                parts=new_parts,
            )
        )
