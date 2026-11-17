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

"""Travel discovery tools with strict schema validation and guided error recovery."""

from __future__ import annotations

import datetime
from typing import Annotated, Any

from pydantic import Field, ValidationError

from app.memory.knowledge_store import get_travel_vector_store
from app.observability.tracing import traced_span
from app.schemas import (
    AdvisoryRetrievalRequest,
    AdvisoryRetrievalResponse,
    AirportLookupRequest,
    AirportLookupResponse,
    AirportRecord,
    FlightOption,
    FlightSearchRequest,
    FlightSearchResponse,
    GuidedToolErrorResponse,
    HotelOption,
    HotelSearchRequest,
    HotelSearchResponse,
)

_VERIFIED_AIRPORT_DIRECTORY: tuple[dict[str, str], ...] = (
    {
        "iata_code": "SFO",
        "airport_name": "San Francisco International Airport",
        "city": "San Francisco",
        "country_code": "US",
        "timezone": "America/Los_Angeles",
    },
    {
        "iata_code": "JFK",
        "airport_name": "John F. Kennedy International Airport",
        "city": "New York",
        "country_code": "US",
        "timezone": "America/New_York",
    },
    {
        "iata_code": "EWR",
        "airport_name": "Newark Liberty International Airport",
        "city": "New York",
        "country_code": "US",
        "timezone": "America/New_York",
    },
    {
        "iata_code": "NRT",
        "airport_name": "Narita International Airport",
        "city": "Tokyo",
        "country_code": "JP",
        "timezone": "Asia/Tokyo",
    },
    {
        "iata_code": "HND",
        "airport_name": "Tokyo Haneda International Airport",
        "city": "Tokyo",
        "country_code": "JP",
        "timezone": "Asia/Tokyo",
    },
    {
        "iata_code": "LHR",
        "airport_name": "London Heathrow Airport",
        "city": "London",
        "country_code": "GB",
        "timezone": "Europe/London",
    },
    {
        "iata_code": "CDG",
        "airport_name": "Paris Charles de Gaulle Airport",
        "city": "Paris",
        "country_code": "FR",
        "timezone": "Europe/Paris",
    },
    {
        "iata_code": "SIN",
        "airport_name": "Singapore Changi Airport",
        "city": "Singapore",
        "country_code": "SG",
        "timezone": "Asia/Singapore",
    },
)

_CURATED_HOTELS_BY_CITY: dict[str, tuple[dict[str, Any], ...]] = {
    "tokyo": (
        {
            "quote_id": "HTL-TYO-101",
            "property_name": "Shinjuku Granbell Boutique Hotel",
            "neighborhood": "Shinjuku",
            "city": "Tokyo",
            "star_rating": 4.2,
            "nightly_rate_usd": 245.0,
            "refundable": True,
            "sustainability_certified": True,
            "amenities": ["high_speed_wifi", "vegetarian_breakfast", "rail_access"],
        },
        {
            "quote_id": "HTL-TYO-204",
            "property_name": "Marunouchi Executive Tower Hotel",
            "neighborhood": "Chiyoda / Tokyo Station",
            "city": "Tokyo",
            "star_rating": 4.8,
            "nightly_rate_usd": 365.0,
            "refundable": True,
            "sustainability_certified": True,
            "amenities": ["business_lounge", "gym", "vegetarian_breakfast"],
        },
    ),
    "london": (
        {
            "quote_id": "HTL-LON-112",
            "property_name": "Bloomsbury Heritage Park Hotel",
            "neighborhood": "Bloomsbury",
            "city": "London",
            "star_rating": 4.3,
            "nightly_rate_usd": 285.0,
            "refundable": True,
            "sustainability_certified": True,
            "amenities": ["high_speed_wifi", "elizabeth_line_walkable", "breakfast"],
        },
        {
            "quote_id": "HTL-LON-309",
            "property_name": "Canary Wharf Riverside Suites",
            "neighborhood": "Canary Wharf",
            "city": "London",
            "star_rating": 4.6,
            "nightly_rate_usd": 340.0,
            "refundable": False,
            "sustainability_certified": True,
            "amenities": ["workspace_desk", "gym", "high_speed_wifi"],
        },
    ),
    "new york": (
        {
            "quote_id": "HTL-NYC-118",
            "property_name": "Midtown Bryant Eco Hotel",
            "neighborhood": "Midtown Manhattan",
            "city": "New York",
            "star_rating": 4.4,
            "nightly_rate_usd": 310.0,
            "refundable": True,
            "sustainability_certified": True,
            "amenities": ["high_speed_wifi", "penn_station_access", "fitness_center"],
        },
    ),
    "paris": (
        {
            "quote_id": "HTL-PAR-140",
            "property_name": "Saint-Germain Left Bank Hotel",
            "neighborhood": "6th Arrondissement",
            "city": "Paris",
            "star_rating": 4.5,
            "nightly_rate_usd": 295.0,
            "refundable": True,
            "sustainability_certified": True,
            "amenities": ["high_speed_wifi", "rer_b_access", "breakfast"],
        },
    ),
    "singapore": (
        {
            "quote_id": "HTL-SIN-155",
            "property_name": "Marina Bay Garden Business Hotel",
            "neighborhood": "Marina Bay",
            "city": "Singapore",
            "star_rating": 4.7,
            "nightly_rate_usd": 320.0,
            "refundable": True,
            "sustainability_certified": True,
            "amenities": ["mrt_access", "high_speed_wifi", "vegetarian_dining"],
        },
    ),
}


@traced_span("tool.lookup_airport_iata_code")
def lookup_airport_iata_code(
    city_or_airport_query: Annotated[
        str,
        Field(
            description="City name, metropolitan region, or airport name to resolve into 3-letter IATA codes."
        ),
    ],
) -> dict[str, Any]:
    """Resolves a city or airport name into verified 3-letter uppercase IATA codes.

    Use this tool before calling `search_verified_flight_options` whenever the
    traveler specifies a city name (for example `'Tokyo'` or `'London'`) instead
    of an explicit IATA code like `'NRT'` or `'LHR'`.

    Args:
        city_or_airport_query: City name, metropolitan region, or partial airport
            name to search in the verified IATA directory.

    Returns:
        A JSON-serializable dictionary matching `AirportLookupResponse` on
        success, or `GuidedToolErrorResponse` with recovery instructions if no
        matching airport exists.
    """
    try:
        req = AirportLookupRequest(city_or_airport_query=city_or_airport_query)
    except ValidationError as exc:
        return GuidedToolErrorResponse(
            error_code="INVALID_AIRPORT_QUERY",
            message=f"Airport query failed schema validation: {exc.errors()[0]['msg']}",
            recoverable=True,
            recovery_instructions=(
                "Pass a non-empty city or airport name of at least 2 characters, "
                "such as 'San Francisco', 'Tokyo', 'London', 'Paris', or 'Singapore'."
            ),
        ).model_dump()

    needle = req.city_or_airport_query.lower()
    matches = [
        AirportRecord(**entry)
        for entry in _VERIFIED_AIRPORT_DIRECTORY
        if needle in entry["city"].lower()
        or needle in entry["airport_name"].lower()
        or needle == entry["iata_code"].lower()
    ]

    if not matches:
        supported_cities = sorted({e["city"] for e in _VERIFIED_AIRPORT_DIRECTORY})
        return GuidedToolErrorResponse(
            error_code="AIRPORT_NOT_FOUND",
            message=f"No verified airport matched '{req.city_or_airport_query}'.",
            recoverable=True,
            recovery_instructions=(
                "Retry with one of the supported hub cities "
                f"({', '.join(supported_cities)}) or ask the traveler for their "
                "3-letter uppercase IATA airport code."
            ),
        ).model_dump()

    return AirportLookupResponse(
        query=req.city_or_airport_query,
        matches=matches,
    ).model_dump()


@traced_span("tool.search_verified_flight_options")
def search_verified_flight_options(
    origin_iata: Annotated[
        str,
        Field(
            pattern=r"^[A-Z]{3}$",
            description="3-letter uppercase IATA code for the origin airport (e.g., 'SFO').",
        ),
    ],
    destination_iata: Annotated[
        str,
        Field(
            pattern=r"^[A-Z]{3}$",
            description="3-letter uppercase IATA code for the destination airport (e.g., 'NRT').",
        ),
    ],
    departure_date: Annotated[
        str,
        Field(
            pattern=r"^\d{4}-\d{2}-\d{2}$",
            description="Departure date in ISO-8601 YYYY-MM-DD format.",
        ),
    ],
    cabin_class: Annotated[
        str,
        Field(
            description="Requested cabin class: 'economy', 'premium_economy', or 'business'."
        ),
    ],
    passengers: Annotated[
        int,
        Field(ge=1, le=9, description="Number of adult passengers traveling (1-9)."),
    ],
    avoid_redeye: Annotated[
        bool,
        Field(
            description="Set True to exclude overnight red-eye departures, or False to include all schedules."
        ),
    ],
) -> dict[str, Any]:
    """Searches verified airline schedules and per-passenger USD fares for a route.

    Validates airport IATA codes, ISO-8601 travel dates, and cabin class against
    `FlightSearchRequest` before querying live route inventory.

    Args:
        origin_iata: 3-letter uppercase IATA departure airport code (e.g., 'SFO').
        destination_iata: 3-letter uppercase IATA arrival airport code (e.g., 'NRT').
        departure_date: Outbound travel date formatted strictly as YYYY-MM-DD.
        cabin_class: Cabin tier ('economy', 'premium_economy', or 'business').
        passengers: Total adult passengers between 1 and 9.
        avoid_redeye: When True, filters out overnight late-night departures.

    Returns:
        A dictionary conforming to `FlightSearchResponse` containing bookable
        `FLT-...` quotes, or `GuidedToolErrorResponse` explaining how to fix
        invalid IATA codes or date formats.
    """
    try:
        req = FlightSearchRequest(
            origin_iata=origin_iata.upper(),
            destination_iata=destination_iata.upper(),
            departure_date=departure_date,
            cabin_class=cabin_class.lower(),  # type: ignore[arg-type]
            passengers=passengers,
            avoid_redeye=avoid_redeye,
        )
    except ValidationError as exc:
        first_err = exc.errors()[0]
        loc = ".".join(str(part) for part in first_err["loc"])
        return GuidedToolErrorResponse(
            error_code="INVALID_FLIGHT_SEARCH_PARAMETERS",
            message=f"Parameter '{loc}' is invalid: {first_err['msg']}",
            recoverable=True,
            recovery_instructions=(
                "Ensure `origin_iata` and `destination_iata` are distinct 3-letter "
                "uppercase IATA codes (call `lookup_airport_iata_code` if you only "
                "have city names), `departure_date` uses YYYY-MM-DD, and `cabin_class` "
                "is one of 'economy', 'premium_economy', or 'business'."
            ),
            suggested_next_tool="lookup_airport_iata_code",
        ).model_dump()

    cabin_multiplier = {
        "economy": 1.0,
        "premium_economy": 1.55,
        "business": 3.1,
    }[req.cabin_class]

    base_route_fare = 890.0
    if {req.origin_iata, req.destination_iata} & {"NRT", "HND", "SIN"}:
        base_route_fare = 1080.0
    elif {req.origin_iata, req.destination_iata} & {"LHR", "CDG"}:
        base_route_fare = 940.0
    elif {req.origin_iata, req.destination_iata} <= {"SFO", "JFK", "EWR"}:
        base_route_fare = 420.0

    daytime_unit_price = round(base_route_fare * cabin_multiplier, 2)
    redeye_unit_price = round(base_route_fare * 0.86 * cabin_multiplier, 2)

    raw_quotes = [
        FlightOption(
            quote_id=f"FLT-{req.origin_iata}{req.destination_iata}-107",
            carrier="All Nippon / Global Partner Air",
            flight_number="NH107",
            origin_iata=req.origin_iata,
            destination_iata=req.destination_iata,
            departure_date=req.departure_date,
            departure_local_time="11:25",
            arrival_local_time="15:10",
            duration_hours=10.75,
            cabin_class=req.cabin_class,
            is_redeye=False,
            refundable=True,
            price_usd_per_passenger=daytime_unit_price,
            total_price_usd=round(daytime_unit_price * req.passengers, 2),
        ),
        FlightOption(
            quote_id=f"FLT-{req.origin_iata}{req.destination_iata}-289",
            carrier="Pacific Star Airlines",
            flight_number="PS289",
            origin_iata=req.origin_iata,
            destination_iata=req.destination_iata,
            departure_date=req.departure_date,
            departure_local_time="23:45",
            arrival_local_time="05:20",
            duration_hours=11.5,
            cabin_class=req.cabin_class,
            is_redeye=True,
            refundable=False,
            price_usd_per_passenger=redeye_unit_price,
            total_price_usd=round(redeye_unit_price * req.passengers, 2),
        ),
    ]

    filtered = [q for q in raw_quotes if not (req.avoid_redeye and q.is_redeye)]
    return FlightSearchResponse(
        origin_iata=req.origin_iata,
        destination_iata=req.destination_iata,
        departure_date=req.departure_date,
        passengers=req.passengers,
        options=filtered,
    ).model_dump()


@traced_span("tool.search_curated_hotel_inventory")
def search_curated_hotel_inventory(
    destination_city: Annotated[
        str,
        Field(description="Destination city name (e.g., 'Tokyo', 'London', 'Paris')."),
    ],
    check_in_date: Annotated[
        str,
        Field(
            pattern=r"^\d{4}-\d{2}-\d{2}$",
            description="Hotel check-in date in YYYY-MM-DD format.",
        ),
    ],
    check_out_date: Annotated[
        str,
        Field(
            pattern=r"^\d{4}-\d{2}-\d{2}$",
            description="Hotel check-out date in YYYY-MM-DD format (must be after check_in_date).",
        ),
    ],
    max_nightly_rate_usd: Annotated[
        float,
        Field(gt=0, le=5000, description="Maximum nightly room rate in USD."),
    ],
) -> dict[str, Any]:
    """Searches vetted hotel properties filtered by stay dates and nightly USD cap.

    Enforces strict date ordering (`check_out_date > check_in_date`) and returns
    sustainability-certified properties with itemized stay totals.

    Args:
        destination_city: Target metropolitan city name (e.g., 'Tokyo').
        check_in_date: Arrival check-in date in ISO-8601 YYYY-MM-DD format.
        check_out_date: Departure check-out date in ISO-8601 YYYY-MM-DD format.
        max_nightly_rate_usd: Maximum acceptable nightly rate in USD before tax.

    Returns:
        A dictionary matching `HotelSearchResponse` with `HTL-...` quotes, or
        `GuidedToolErrorResponse` with instructions to adjust dates or budget.
    """
    try:
        req = HotelSearchRequest(
            destination_city=destination_city,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            max_nightly_rate_usd=max_nightly_rate_usd,
        )
    except ValidationError as exc:
        first_err = exc.errors()[0]
        return GuidedToolErrorResponse(
            error_code="INVALID_HOTEL_SEARCH_PARAMETERS",
            message=f"Hotel search validation failed: {first_err['msg']}",
            recoverable=True,
            recovery_instructions=(
                "Verify both `check_in_date` and `check_out_date` use YYYY-MM-DD, "
                "ensure `check_out_date` is at least 1 day after `check_in_date`, "
                "and pass a positive `max_nightly_rate_usd` <= 5000."
            ),
        ).model_dump()

    nights = (
        datetime.date.fromisoformat(req.check_out_date)
        - datetime.date.fromisoformat(req.check_in_date)
    ).days

    city_key = req.destination_city.lower().strip()
    matched_inventory: tuple[dict[str, Any], ...] = ()
    for known_city, props in _CURATED_HOTELS_BY_CITY.items():
        if known_city in city_key or city_key in known_city:
            matched_inventory = props
            break

    if not matched_inventory:
        supported = ", ".join(title.title() for title in _CURATED_HOTELS_BY_CITY)
        return GuidedToolErrorResponse(
            error_code="UNSUPPORTED_HOTEL_MARKET",
            message=f"No curated hotel inventory is indexed for '{req.destination_city}'.",
            recoverable=True,
            recovery_instructions=(
                f"Query one of the supported metropolitan markets ({supported}) "
                "or use the nearest primary gateway city."
            ),
        ).model_dump()

    affordable = [
        HotelOption(
            **prop,
            nights=nights,
            total_stay_usd=round(float(prop["nightly_rate_usd"]) * nights, 2),
        )
        for prop in matched_inventory
        if float(prop["nightly_rate_usd"]) <= req.max_nightly_rate_usd
    ]

    if not affordable:
        lowest = min(float(p["nightly_rate_usd"]) for p in matched_inventory)
        return GuidedToolErrorResponse(
            error_code="NIGHTLY_RATE_CAP_TOO_LOW",
            message=(
                f"No properties in {req.destination_city} matched <= "
                f"${req.max_nightly_rate_usd:.2f}/night (lowest available is ${lowest:.2f}/night)."
            ),
            recoverable=True,
            recovery_instructions=(
                f"Retry `search_curated_hotel_inventory` with `max_nightly_rate_usd` "
                f"set to at least {lowest:.2f} or shorten the stay duration."
            ),
        ).model_dump()

    return HotelSearchResponse(
        destination_city=req.destination_city,
        nights=nights,
        options=affordable,
    ).model_dump()


@traced_span("tool.retrieve_destination_travel_advisories")
def retrieve_destination_travel_advisories(
    destination_query: Annotated[
        str,
        Field(
            description="Destination city, country, or travel policy query (e.g., 'Tokyo Japan visa and transit')."
        ),
    ],
    top_k: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Maximum number of ranked advisory documents to retrieve (1-5).",
        ),
    ],
) -> dict[str, Any]:
    """Retrieves destination safety tiers, visa rules, transit tips, and per-diems.

    Executes semantic vector similarity search over the persistent travel policy
    database (`PersistentTravelVectorStore`) to ground itinerary planning.

    Args:
        destination_query: Destination city, country name, or policy topic.
        top_k: Maximum number of relevant advisory records to return (1 to 5).

    Returns:
        A dictionary conforming to `AdvisoryRetrievalResponse`, or a structured
        `GuidedToolErrorResponse` listing indexed destinations if none matched.
    """
    try:
        req = AdvisoryRetrievalRequest(
            destination_query=destination_query, top_k=top_k
        )
    except ValidationError as exc:
        return GuidedToolErrorResponse(
            error_code="INVALID_ADVISORY_QUERY",
            message=f"Advisory retrieval query invalid: {exc.errors()[0]['msg']}",
            recoverable=True,
            recovery_instructions=(
                "Provide a destination query of at least 2 characters and set "
                "`top_k` between 1 and 5."
            ),
        ).model_dump()

    store = get_travel_vector_store()
    hits = store.search(req.destination_query, top_k=req.top_k)
    if not hits:
        indexed = ", ".join(store.list_available_destinations())
        return GuidedToolErrorResponse(
            error_code="NO_MATCHING_ADVISORY_DOCUMENT",
            message=f"No travel advisory matched '{req.destination_query}'.",
            recoverable=True,
            recovery_instructions=(
                f"Retry `retrieve_destination_travel_advisories` using a broader "
                f"country or city keyword from the indexed catalog: {indexed}."
            ),
        ).model_dump()

    return AdvisoryRetrievalResponse(
        query=req.destination_query,
        results=hits,
    ).model_dump()
