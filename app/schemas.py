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

"""Strict Pydantic v2 JSON schemas for tool inputs, tool outputs, and self-evals.

Every model enforces `extra="forbid"` and explicit field constraints so LLM
tool arguments and structured agent outputs are deterministically validated.
"""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictSchemaModel(BaseModel):
    """Base schema rejecting undeclared properties and stripping whitespace."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GuidedToolErrorResponse(StrictSchemaModel):
    """Structured error contract guiding the LLM toward autonomous recovery."""

    status: Literal["error"] = Field(
        default="error",
        description="Indicates the tool call did not complete and requires adjustment.",
    )
    error_code: str = Field(
        min_length=3,
        max_length=64,
        description="Machine-readable uppercase error code categorizing the failure.",
    )
    message: str = Field(
        min_length=5,
        description="Human-readable explanation of why the input or operation failed.",
    )
    recoverable: bool = Field(
        description="True when the agent can retry with corrected arguments or a prerequisite tool.",
    )
    recovery_instructions: str = Field(
        min_length=10,
        description="Step-by-step instructions telling the LLM how to recover.",
    )
    suggested_next_tool: str | None = Field(
        default=None,
        description="Optional specific tool name the agent should invoke next.",
    )


class AirportLookupRequest(StrictSchemaModel):
    """Validated input schema for resolving a city or airport name to IATA codes."""

    city_or_airport_query: str = Field(
        min_length=2,
        max_length=100,
        description="City name, metropolitan region, or airport name (e.g., 'Tokyo' or 'Heathrow').",
    )


class AirportRecord(StrictSchemaModel):
    """Single verified IATA airport descriptor."""

    iata_code: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="Standard 3-letter uppercase IATA airport code.",
    )
    airport_name: str = Field(description="Official international airport name.")
    city: str = Field(description="Metropolitan area served by the airport.")
    country_code: str = Field(
        pattern=r"^[A-Z]{2}$",
        description="ISO 3166-1 alpha-2 country code.",
    )
    timezone: str = Field(description="IANA timezone identifier.")


class AirportLookupResponse(StrictSchemaModel):
    """Validated output schema for `lookup_airport_iata_code`."""

    status: Literal["success"] = "success"
    query: str
    matches: list[AirportRecord]


class FlightSearchRequest(StrictSchemaModel):
    """Validated input schema for `search_verified_flight_options`."""

    origin_iata: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="3-letter uppercase IATA code for the departure airport (e.g., 'SFO').",
    )
    destination_iata: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="3-letter uppercase IATA code for the arrival airport (e.g., 'NRT').",
    )
    departure_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Outbound departure date in strict ISO-8601 YYYY-MM-DD format.",
    )
    cabin_class: Literal["economy", "premium_economy", "business"] = Field(
        description="Requested airline cabin class.",
    )
    passengers: int = Field(
        ge=1,
        le=9,
        description="Number of adult travelers (1 to 9).",
    )
    avoid_redeye: bool = Field(
        default=False,
        description="When True, filters out overnight red-eye departures.",
    )

    @field_validator("departure_date")
    @classmethod
    def _validate_iso_date(cls, value: str) -> str:
        datetime.date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _validate_distinct_airports(self) -> FlightSearchRequest:
        if self.origin_iata == self.destination_iata:
            raise ValueError(
                "origin_iata and destination_iata must be different airports."
            )
        return self


class FlightOption(StrictSchemaModel):
    """Verified flight quote returned by inventory search."""

    quote_id: str = Field(description="Unique bookable flight quote identifier.")
    carrier: str = Field(description="Operating airline name.")
    flight_number: str = Field(description="IATA carrier + flight number.")
    origin_iata: str = Field(pattern=r"^[A-Z]{3}$")
    destination_iata: str = Field(pattern=r"^[A-Z]{3}$")
    departure_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    departure_local_time: str = Field(description="24-hour local departure HH:MM.")
    arrival_local_time: str = Field(description="24-hour local arrival HH:MM.")
    duration_hours: float = Field(gt=0)
    cabin_class: Literal["economy", "premium_economy", "business"]
    is_redeye: bool
    refundable: bool
    price_usd_per_passenger: float = Field(gt=0)
    total_price_usd: float = Field(gt=0)


class FlightSearchResponse(StrictSchemaModel):
    """Validated output schema for `search_verified_flight_options`."""

    status: Literal["success"] = "success"
    origin_iata: str
    destination_iata: str
    departure_date: str
    passengers: int
    options: list[FlightOption]


class HotelSearchRequest(StrictSchemaModel):
    """Validated input schema for `search_curated_hotel_inventory`."""

    destination_city: str = Field(
        min_length=2,
        max_length=80,
        description="Destination city name (e.g., 'Tokyo', 'London', 'New York').",
    )
    check_in_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Check-in date in YYYY-MM-DD format.",
    )
    check_out_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Check-out date in YYYY-MM-DD format (must be after check_in_date).",
    )
    max_nightly_rate_usd: float = Field(
        gt=0,
        le=5000,
        description="Maximum nightly room rate ceiling in USD.",
    )

    @model_validator(mode="after")
    def _validate_stay_window(self) -> HotelSearchRequest:
        start = datetime.date.fromisoformat(self.check_in_date)
        end = datetime.date.fromisoformat(self.check_out_date)
        if end <= start:
            raise ValueError("check_out_date must be strictly after check_in_date.")
        if (end - start).days > 30:
            raise ValueError(
                "Single hotel booking window cannot exceed 30 nights; split longer stays."
            )
        return self


class HotelOption(StrictSchemaModel):
    """Curated hotel property quote."""

    quote_id: str = Field(description="Unique bookable hotel quote identifier.")
    property_name: str
    neighborhood: str
    city: str
    star_rating: float = Field(ge=1.0, le=5.0)
    nightly_rate_usd: float = Field(gt=0)
    nights: int = Field(ge=1)
    total_stay_usd: float = Field(gt=0)
    refundable: bool
    sustainability_certified: bool
    amenities: list[str]


class HotelSearchResponse(StrictSchemaModel):
    """Validated output schema for `search_curated_hotel_inventory`."""

    status: Literal["success"] = "success"
    destination_city: str
    nights: int
    options: list[HotelOption]


class AdvisoryRetrievalRequest(StrictSchemaModel):
    """Validated input schema for `retrieve_destination_travel_advisories`."""

    destination_query: str = Field(
        min_length=2,
        max_length=120,
        description="Destination city, country, or travel policy topic to retrieve.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of knowledge-base documents to return.",
    )


class AdvisoryDocumentHit(StrictSchemaModel):
    """Single retrieved advisory or policy document with similarity score."""

    doc_id: str
    destination: str
    country_code: str
    safety_tier: Literal["level_1_normal", "level_2_caution", "level_3_reconsider"]
    daily_per_diem_usd: float = Field(gt=0)
    visa_and_entry_rules: str
    local_transit_guidance: str
    expense_policy_notes: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class AdvisoryRetrievalResponse(StrictSchemaModel):
    """Validated output schema for `retrieve_destination_travel_advisories`."""

    status: Literal["success"] = "success"
    query: str
    results: list[AdvisoryDocumentHit]


class BudgetCalculationRequest(StrictSchemaModel):
    """Validated input schema for `calculate_trip_budget_breakdown`."""

    flight_total_usd: float = Field(ge=0, description="Total airfare cost in USD.")
    hotel_total_usd: float = Field(ge=0, description="Total lodging cost in USD.")
    trip_days: int = Field(
        ge=1, le=60, description="Total number of trip days for per-diem calculation."
    )
    daily_per_diem_usd: float = Field(
        gt=0, le=1000, description="Estimated daily meals and local transit per diem."
    )
    target_budget_usd: float = Field(
        gt=0, description="Traveler's overall trip budget cap in USD."
    )


class BudgetCalculationResponse(StrictSchemaModel):
    """Validated output schema for `calculate_trip_budget_breakdown`."""

    status: Literal["success"] = "success"
    flight_total_usd: float
    hotel_total_usd: float
    per_diem_total_usd: float
    estimated_taxes_and_fees_usd: float
    grand_total_usd: float
    target_budget_usd: float
    within_budget: bool
    remaining_or_overage_usd: float
    optimization_recommendations: list[str]


class ItineraryBookingRequest(StrictSchemaModel):
    """Validated input schema for high-stakes `reserve_itinerary_booking`."""

    flight_quote_id: str = Field(
        pattern=r"^FLT-[A-Z0-9-]+$",
        description="Verified flight quote ID from `search_verified_flight_options`.",
    )
    hotel_quote_id: str = Field(
        pattern=r"^HTL-[A-Z0-9-]+$",
        description="Verified hotel quote ID from `search_curated_hotel_inventory`.",
    )
    traveler_full_name: str = Field(
        min_length=2,
        max_length=100,
        description="Primary traveler's legal name matching travel documents.",
    )
    total_authorized_usd: float = Field(
        gt=0,
        le=25000,
        description="Total USD amount authorized by the traveler for this reservation.",
    )
    trip_purpose: str = Field(
        min_length=3,
        max_length=200,
        description="Concise business or personal purpose of travel.",
    )


class ItineraryBookingResponse(StrictSchemaModel):
    """Validated output schema for `reserve_itinerary_booking`."""

    status: Literal["confirmed"] = "confirmed"
    booking_reference: str
    flight_quote_id: str
    hotel_quote_id: str
    charged_amount_usd: float
    human_approval_verified: bool = True
    cancellation_policy_summary: str


class ExpenseClaimRequest(StrictSchemaModel):
    """Validated input schema for `submit_expense_reimbursement_claim`."""

    trip_reference: str = Field(
        min_length=4,
        max_length=40,
        description="Booking reference or trip identifier (e.g., 'TRV-2026-8841').",
    )
    expense_category: Literal[
        "lodging", "airfare", "meals", "ground_transport", "conference_fee"
    ] = Field(description="Standardized travel expense category.")
    amount_usd: float = Field(
        gt=0,
        le=10000,
        description="Reimbursement amount requested in USD.",
    )
    merchant_name: str = Field(
        min_length=2,
        max_length=120,
        description="Merchant or vendor name appearing on the itemized receipt.",
    )
    expense_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Receipt transaction date in YYYY-MM-DD format.",
    )
    receipt_notes: str = Field(
        default="",
        max_length=500,
        description="Optional business justification or receipt summary (auto-redacted for PII).",
    )

    @field_validator("expense_date")
    @classmethod
    def _validate_receipt_date(cls, value: str) -> str:
        datetime.date.fromisoformat(value)
        return value


class ExpenseClaimResponse(StrictSchemaModel):
    """Validated output schema for `submit_expense_reimbursement_claim`."""

    status: Literal["submitted"] = "submitted"
    claim_id: str
    trip_reference: str
    expense_category: str
    approved_amount_usd: float
    policy_compliance_note: str
    pii_redacted: bool


class ItineraryAuditVerdict(StrictSchemaModel):
    """Structured `output_schema` emitted by `policy_and_budget_auditor`."""

    approved: bool = Field(
        description="True only when the proposed itinerary satisfies budget, safety, and completeness rules.",
    )
    budget_compliant: bool = Field(
        description="Whether total estimated cost is within the traveler's stated ceiling.",
    )
    safety_advisory_checked: bool = Field(
        description="Whether destination entry, safety, and local transit advisories are included.",
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="Specific actionable deficiencies that `itinerary_refiner` must fix if not approved.",
    )
    auditor_summary: str = Field(
        description="Concise rationale summarizing the quality and policy audit outcome.",
    )
