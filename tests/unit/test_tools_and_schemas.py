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

"""Unit tests for Category 1: Tool & Interface Design (docstrings, naming, schemas, error recovery)."""

from __future__ import annotations

import inspect

import pytest
from google.adk.tools import FunctionTool
from pydantic import ValidationError

from app.schemas import FlightSearchRequest, ItineraryAuditVerdict
from app.tools import (
    TOOL_JSON_SCHEMAS,
    calculate_trip_budget_breakdown,
    lookup_airport_iata_code,
    reserve_itinerary_booking,
    retrieve_destination_travel_advisories,
    search_curated_hotel_inventory,
    search_verified_flight_options,
    submit_expense_reimbursement_claim,
)

ALL_DOMAIN_TOOL_FUNCTIONS = (
    lookup_airport_iata_code,
    search_verified_flight_options,
    search_curated_hotel_inventory,
    retrieve_destination_travel_advisories,
    calculate_trip_budget_breakdown,
    reserve_itinerary_booking,
    submit_expense_reimbursement_claim,
)


def test_all_tools_have_comprehensive_docstrings_and_specific_names() -> None:
    """Verifies specific action-oriented naming and complete parameter docstrings."""
    for func in ALL_DOMAIN_TOOL_FUNCTIONS:
        assert "_" in func.__name__, f"Tool {func.__name__} must use descriptive verb_noun naming"
        assert len(func.__name__) >= 18, f"Tool name {func.__name__} is too generic"

        doc = inspect.getdoc(func)
        assert doc is not None and len(doc) >= 120
        assert "Args:" in doc
        assert "Returns:" in doc

        sig = inspect.signature(func)
        for param_name in sig.parameters:
            assert param_name in doc, (
                f"Parameter '{param_name}' missing from {func.__name__} docstring"
            )


def test_explicit_json_schemas_enforce_strict_constraints() -> None:
    """Verifies JSON Schemas disallow extra fields and constrain types/patterns."""
    assert len(TOOL_JSON_SCHEMAS) == len(ALL_DOMAIN_TOOL_FUNCTIONS)
    for name, schema in TOOL_JSON_SCHEMAS.items():
        assert schema.get("additionalProperties") is False, (
            f"Schema for {name} must forbid undeclared extra fields"
        )
        assert "properties" in schema and "required" in schema

    # ADK FunctionTool declaration carries parameter JSON schema constraints.
    decl = FunctionTool(search_verified_flight_options)._get_declaration()
    assert decl is not None
    assert decl.parameters_json_schema is not None
    props = decl.parameters_json_schema["properties"]
    assert props["origin_iata"]["pattern"] == "^[A-Z]{3}$"
    assert props["passengers"]["minimum"] == 1
    assert props["passengers"]["maximum"] == 9

    # Undeclared extra fields are strictly rejected.
    with pytest.raises(ValidationError):
        FlightSearchRequest(
            origin_iata="SFO",
            destination_iata="NRT",
            departure_date="2026-11-20",
            cabin_class="economy",
            passengers=1,
            undeclared_field="not_allowed",  # type: ignore[call-arg]
        )

    verdict = ItineraryAuditVerdict(
        approved=True,
        budget_compliant=True,
        safety_advisory_checked=True,
        issues_found=[],
        auditor_summary="All travel policy checks passed.",
    )
    assert verdict.approved is True


def test_guided_error_handling_provides_actionable_recovery_instructions() -> None:
    """Verifies invalid tool inputs return structured recovery instructions without crashing."""
    # 1. City name instead of 3-letter IATA code guides LLM to lookup_airport_iata_code.
    bad_flight = search_verified_flight_options(
        origin_iata="San Francisco",
        destination_iata="Tokyo",
        departure_date="2026-11-20",
        cabin_class="economy",
        passengers=1,
        avoid_redeye=True,
    )
    assert bad_flight["status"] == "error"
    assert bad_flight["error_code"] == "INVALID_FLIGHT_SEARCH_PARAMETERS"
    assert bad_flight["recoverable"] is True
    assert bad_flight["suggested_next_tool"] == "lookup_airport_iata_code"
    assert "lookup_airport_iata_code" in bad_flight["recovery_instructions"]

    # 2. Check-out date before check-in date returns guided date fix instructions.
    bad_hotel = search_curated_hotel_inventory(
        destination_city="Tokyo",
        check_in_date="2026-11-25",
        check_out_date="2026-11-20",
        max_nightly_rate_usd=300.0,
    )
    assert bad_hotel["status"] == "error"
    assert bad_hotel["error_code"] == "INVALID_HOTEL_SEARCH_PARAMETERS"
    assert bad_hotel["recoverable"] is True
    assert "check_out_date" in bad_hotel["recovery_instructions"]

    # 3. Unrealistically low nightly rate informs LLM of exact minimum available rate.
    low_cap_hotel = search_curated_hotel_inventory(
        destination_city="Tokyo",
        check_in_date="2026-11-20",
        check_out_date="2026-11-24",
        max_nightly_rate_usd=50.0,
    )
    assert low_cap_hotel["status"] == "error"
    assert low_cap_hotel["error_code"] == "NIGHTLY_RATE_CAP_TOO_LOW"
    assert "245.00" in low_cap_hotel["recovery_instructions"]

    # 4. Malformed booking quote ID guides LLM back to flight/hotel search tools.
    bad_booking = reserve_itinerary_booking(
        flight_quote_id="INVALID-999",
        hotel_quote_id="HTL-TYO-101",
        traveler_full_name="Alex Rivera",
        total_authorized_usd=1800.0,
        trip_purpose="Customer summit",
    )
    assert bad_booking["status"] == "error"
    assert bad_booking["error_code"] == "INVALID_BOOKING_PAYLOAD"
    assert bad_booking["suggested_next_tool"] == "search_verified_flight_options"


def test_happy_path_discovery_and_budget_tools() -> None:
    """Verifies happy-path return contracts across discovery and budget tools."""
    airports = lookup_airport_iata_code("Tokyo")
    assert airports["status"] == "success"
    codes = {m["iata_code"] for m in airports["matches"]}
    assert {"NRT", "HND"} <= codes

    flights = search_verified_flight_options(
        origin_iata="SFO",
        destination_iata="NRT",
        departure_date="2026-11-20",
        cabin_class="economy",
        passengers=1,
        avoid_redeye=True,
    )
    assert flights["status"] == "success"
    assert all(not opt["is_redeye"] for opt in flights["options"])

    hotels = search_curated_hotel_inventory(
        destination_city="Tokyo",
        check_in_date="2026-11-20",
        check_out_date="2026-11-24",
        max_nightly_rate_usd=300.0,
    )
    assert hotels["status"] == "success"
    assert hotels["nights"] == 4

    budget = calculate_trip_budget_breakdown(
        flight_total_usd=1080.0,
        hotel_total_usd=980.0,
        trip_days=4,
        daily_per_diem_usd=145.0,
        target_budget_usd=3200.0,
    )
    assert budget["status"] == "success"
    assert budget["within_budget"] is True
    assert budget["grand_total_usd"] < 3200.0
