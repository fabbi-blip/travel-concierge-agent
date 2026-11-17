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

"""Budget calculation and Human-in-the-Loop (HITL) booking and expense tools."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from google.adk.tools import FunctionTool
from pydantic import Field, ValidationError

from app.observability.tracing import traced_span
from app.schemas import (
    AdvisoryRetrievalRequest,
    AirportLookupRequest,
    BudgetCalculationRequest,
    BudgetCalculationResponse,
    ExpenseClaimRequest,
    ExpenseClaimResponse,
    FlightSearchRequest,
    GuidedToolErrorResponse,
    HotelSearchRequest,
    ItineraryBookingRequest,
    ItineraryBookingResponse,
)
from app.security.pii_redactor import get_pii_redactor


@traced_span("tool.calculate_trip_budget_breakdown")
def calculate_trip_budget_breakdown(
    flight_total_usd: Annotated[
        float, Field(ge=0, description="Selected flight total cost in USD.")
    ],
    hotel_total_usd: Annotated[
        float, Field(ge=0, description="Selected hotel stay pre-tax total in USD.")
    ],
    trip_days: Annotated[
        int, Field(ge=1, le=60, description="Total number of trip days (1-60).")
    ],
    daily_per_diem_usd: Annotated[
        float,
        Field(
            gt=0,
            le=1000,
            description="Destination daily meals and local transit per-diem in USD.",
        ),
    ],
    target_budget_usd: Annotated[
        float,
        Field(gt=0, description="Traveler's total trip budget ceiling in USD."),
    ],
) -> dict[str, Any]:
    """Calculates itemized trip costs including lodging taxes and daily per-diem.

    Compares the grand total against `target_budget_usd` and generates concrete
    cost-reduction steps whenever a proposed itinerary exceeds the budget cap.

    Args:
        flight_total_usd: Total round-trip or outbound airfare in USD.
        hotel_total_usd: Pre-tax accommodation total across all nights in USD.
        trip_days: Total duration of the trip in days for per-diem allowance.
        daily_per_diem_usd: Daily allowance for meals and local transit in USD.
        target_budget_usd: Maximum overall budget ceiling specified by traveler.

    Returns:
        A dictionary matching `BudgetCalculationResponse` with itemized totals
        and `optimization_recommendations`, or `GuidedToolErrorResponse` if
        numeric inputs are out of range.
    """
    try:
        req = BudgetCalculationRequest(
            flight_total_usd=flight_total_usd,
            hotel_total_usd=hotel_total_usd,
            trip_days=trip_days,
            daily_per_diem_usd=daily_per_diem_usd,
            target_budget_usd=target_budget_usd,
        )
    except ValidationError as exc:
        return GuidedToolErrorResponse(
            error_code="INVALID_BUDGET_INPUTS",
            message=f"Budget calculation inputs failed validation: {exc.errors()[0]['msg']}",
            recoverable=True,
            recovery_instructions=(
                "Provide non-negative flight/hotel USD totals, `trip_days` between "
                "1 and 60, and positive `daily_per_diem_usd` and `target_budget_usd`."
            ),
        ).model_dump()

    taxes_and_fees = round(req.hotel_total_usd * 0.14 + req.flight_total_usd * 0.05, 2)
    per_diem_total = round(req.trip_days * req.daily_per_diem_usd, 2)
    grand_total = round(
        req.flight_total_usd + req.hotel_total_usd + taxes_and_fees + per_diem_total, 2
    )
    delta = round(req.target_budget_usd - grand_total, 2)
    within_budget = grand_total <= req.target_budget_usd

    recommendations: list[str] = []
    if within_budget:
        recommendations.append(
            f"Itinerary is within budget with ${delta:,.2f} USD contingency remaining."
        )
    else:
        overage = abs(delta)
        recommendations.append(
            f"Itinerary exceeds target budget by ${overage:,.2f} USD."
        )
        if req.hotel_total_usd > 0:
            recommendations.append(
                "Switch to a boutique neighborhood hotel (e.g., <= $250/night) "
                "to reduce lodging and 14% occupancy tax."
            )
        if req.flight_total_usd > 900:
            recommendations.append(
                "Evaluate economy cabin or off-peak departure quotes via "
                "`search_verified_flight_options`."
            )

    return BudgetCalculationResponse(
        flight_total_usd=round(req.flight_total_usd, 2),
        hotel_total_usd=round(req.hotel_total_usd, 2),
        per_diem_total_usd=per_diem_total,
        estimated_taxes_and_fees_usd=taxes_and_fees,
        grand_total_usd=grand_total,
        target_budget_usd=round(req.target_budget_usd, 2),
        within_budget=within_budget,
        remaining_or_overage_usd=delta,
        optimization_recommendations=recommendations,
    ).model_dump()


def requires_booking_confirmation(
    flight_quote_id: str = "",
    hotel_quote_id: str = "",
    traveler_full_name: str = "",
    total_authorized_usd: float = 0.0,
    trip_purpose: str = "",
) -> bool:
    """Human-in-the-Loop predicate pausing execution before financial commitment."""
    _ = (flight_quote_id, hotel_quote_id, traveler_full_name, trip_purpose)
    return float(total_authorized_usd) > 0.0


@traced_span("tool.reserve_itinerary_booking")
def reserve_itinerary_booking(
    flight_quote_id: Annotated[
        str,
        Field(
            pattern=r"^FLT-[A-Z0-9-]+$",
            description="Flight quote ID starting with 'FLT-' from `search_verified_flight_options`.",
        ),
    ],
    hotel_quote_id: Annotated[
        str,
        Field(
            pattern=r"^HTL-[A-Z0-9-]+$",
            description="Hotel quote ID starting with 'HTL-' from `search_curated_hotel_inventory`.",
        ),
    ],
    traveler_full_name: Annotated[
        str,
        Field(description="Traveler's legal full name matching travel ID."),
    ],
    total_authorized_usd: Annotated[
        float,
        Field(
            gt=0,
            le=25000,
            description="Total USD amount explicitly authorized by the traveler.",
        ),
    ],
    trip_purpose: Annotated[
        str,
        Field(description="Brief business or personal purpose for the booking audit log."),
    ],
) -> dict[str, Any]:
    """Executes a confirmed flight and hotel reservation after human approval.

    Protected by an explicit ADK Human-in-the-Loop confirmation gate
    (`require_confirmation=requires_booking_confirmation`) so funds are never
    committed without user sign-off.

    Args:
        flight_quote_id: Verified flight quote identifier (`FLT-...`).
        hotel_quote_id: Verified hotel quote identifier (`HTL-...`).
        traveler_full_name: Full name of the primary traveler.
        total_authorized_usd: Total USD charge approved by the user.
        trip_purpose: Concise trip reason recorded on the itinerary folio.

    Returns:
        A dictionary conforming to `ItineraryBookingResponse` with a `TRV-...`
        booking reference, or `GuidedToolErrorResponse` if quote IDs are malformed.
    """
    try:
        req = ItineraryBookingRequest(
            flight_quote_id=flight_quote_id,
            hotel_quote_id=hotel_quote_id,
            traveler_full_name=traveler_full_name,
            total_authorized_usd=total_authorized_usd,
            trip_purpose=trip_purpose,
        )
    except ValidationError as exc:
        first_err = exc.errors()[0]
        return GuidedToolErrorResponse(
            error_code="INVALID_BOOKING_PAYLOAD",
            message=f"Booking validation failed: {first_err['msg']}",
            recoverable=True,
            recovery_instructions=(
                "Pass valid `FLT-...` and `HTL-...` quote IDs obtained from "
                "`search_verified_flight_options` and `search_curated_hotel_inventory`, "
                "along with a positive `total_authorized_usd`."
            ),
            suggested_next_tool="search_verified_flight_options",
        ).model_dump()

    redactor = get_pii_redactor()
    _ = redactor.redact_text(req.trip_purpose)
    booking_ref = f"TRV-2026-{uuid.uuid4().hex[:6].upper()}"

    return ItineraryBookingResponse(
        booking_reference=booking_ref,
        flight_quote_id=req.flight_quote_id,
        hotel_quote_id=req.hotel_quote_id,
        charged_amount_usd=round(req.total_authorized_usd, 2),
        human_approval_verified=True,
        cancellation_policy_summary=(
            "Free modification or cancellation within 24 hours of issuance; "
            "fare/property rules apply thereafter."
        ),
    ).model_dump()


@traced_span("tool.submit_expense_reimbursement_claim")
def submit_expense_reimbursement_claim(
    trip_reference: Annotated[
        str,
        Field(description="Booking reference or trip ID (e.g., 'TRV-2026-A1B2C3')."),
    ],
    expense_category: Annotated[
        str,
        Field(
            description="Expense category: 'lodging', 'airfare', 'meals', 'ground_transport', or 'conference_fee'."
        ),
    ],
    amount_usd: Annotated[
        float,
        Field(gt=0, le=10000, description="Itemized receipt total in USD."),
    ],
    merchant_name: Annotated[
        str,
        Field(description="Vendor or merchant name printed on the receipt."),
    ],
    expense_date: Annotated[
        str,
        Field(
            pattern=r"^\d{4}-\d{2}-\d{2}$",
            description="Receipt date in ISO-8601 YYYY-MM-DD format.",
        ),
    ],
    receipt_notes: Annotated[
        str,
        Field(
            description="Business justification or receipt description (PII is automatically scrubbed)."
        ),
    ],
) -> dict[str, Any]:
    """Submits an itemized post-trip expense reimbursement claim after HITL review.

    Protected by `FunctionTool(..., require_confirmation=True)` and automatically
    scrubs any sensitive PII (card numbers, emails, phone numbers, SSNs) from
    `receipt_notes` prior to persistence.

    Args:
        trip_reference: Associated itinerary reference (`TRV-...`).
        expense_category: One of `'lodging'`, `'airfare'`, `'meals'`,
            `'ground_transport'`, or `'conference_fee'`.
        amount_usd: Requested reimbursement amount in USD.
        merchant_name: Official merchant name from the receipt.
        expense_date: Transaction date formatted as YYYY-MM-DD.
        receipt_notes: Context or justification notes for finance audit.

    Returns:
        A dictionary matching `ExpenseClaimResponse` containing the `EXP-...`
        claim identifier, or `GuidedToolErrorResponse` with remediation guidance.
    """
    try:
        req = ExpenseClaimRequest(
            trip_reference=trip_reference,
            expense_category=expense_category.lower(),  # type: ignore[arg-type]
            amount_usd=amount_usd,
            merchant_name=merchant_name,
            expense_date=expense_date,
            receipt_notes=receipt_notes,
        )
    except ValidationError as exc:
        first_err = exc.errors()[0]
        return GuidedToolErrorResponse(
            error_code="INVALID_EXPENSE_CLAIM",
            message=f"Expense claim validation failed: {first_err['msg']}",
            recoverable=True,
            recovery_instructions=(
                "Ensure `expense_category` is one of 'lodging', 'airfare', 'meals', "
                "'ground_transport', or 'conference_fee', `expense_date` is YYYY-MM-DD, "
                "and `amount_usd` is > 0."
            ),
        ).model_dump()

    redactor = get_pii_redactor()
    was_redacted = redactor.contains_pii(req.receipt_notes) or redactor.contains_pii(
        req.merchant_name
    )
    _ = redactor.redact_text(req.receipt_notes)

    claim_id = f"EXP-2026-{uuid.uuid4().hex[:6].upper()}"
    policy_note = "Compliant with standard concierge reimbursement guidelines."
    if req.expense_category == "meals" and req.amount_usd > 120.0:
        policy_note = (
            "Queued for itemized attendee review because meal receipt exceeds $120 USD."
        )

    return ExpenseClaimResponse(
        claim_id=claim_id,
        trip_reference=req.trip_reference,
        expense_category=req.expense_category,
        approved_amount_usd=round(req.amount_usd, 2),
        policy_compliance_note=policy_note,
        pii_redacted=was_redacted,
    ).model_dump()


# High-stakes tools wrapped with explicit Human-in-the-Loop confirmation hooks.
reserve_itinerary_booking_tool = FunctionTool(
    reserve_itinerary_booking,
    require_confirmation=requires_booking_confirmation,
)

submit_expense_reimbursement_claim_tool = FunctionTool(
    submit_expense_reimbursement_claim,
    require_confirmation=True,
)

# Registry of explicit JSON Schema (Draft 2020-12) definitions for validation & inspection.
TOOL_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "lookup_airport_iata_code": AirportLookupRequest.model_json_schema(),
    "search_verified_flight_options": FlightSearchRequest.model_json_schema(),
    "search_curated_hotel_inventory": HotelSearchRequest.model_json_schema(),
    "retrieve_destination_travel_advisories": AdvisoryRetrievalRequest.model_json_schema(),
    "calculate_trip_budget_breakdown": BudgetCalculationRequest.model_json_schema(),
    "reserve_itinerary_booking": ItineraryBookingRequest.model_json_schema(),
    "submit_expense_reimbursement_claim": ExpenseClaimRequest.model_json_schema(),
}
