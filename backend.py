import os
import re
import certifi
from dotenv import load_dotenv


load_dotenv()


os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt


from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)


from langchain_groq import ChatGroq


from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================


def get_database_url():

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:

        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:

        separator = (
            "&"
            if "?" in database_url
            else "?"
        )

        database_url = (
            f"{database_url}"
            f"{separator}"
            "sslmode=require"
        )

    return database_url


# ============================================================
# GROQ CONFIGURATION
# ============================================================


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )


# ============================================================
# LLM
# ============================================================


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
)


# ============================================================
# TRAVEL STATE
# ============================================================


class TravelState(
    TypedDict,
    total=False
):

    messages: Annotated[
        list[AnyMessage],
        operator.add
    ]

    user_query: str

    # Supervisor + guardrail state

    guardrail_allowed: bool

    guardrail_reason: str

    selected_agents: list[str]

    trip_constraints: dict[str, Any]

    supervisor_reasoning: str

    # Specialist results

    flight_results: str

    hotel_results: str

    weather_results: str

    itinerary: str

    # Budget + HITL

    budget_results: str

    approval_request: str

    approved: bool

    human_feedback: str

    final_response: str

    # Tracking

    llm_calls: int


# ============================================================
# AGENT CONFIGURATION
# ============================================================


KNOWN_AGENTS = {

    "flight_agent",

    "hotel_agent",

    "weather_agent",

    "budget_agent",

    "itinerary_agent",

}


AGENT_ORDER = [

    "flight_agent",

    "hotel_agent",

    "weather_agent",

    "budget_agent",

    "itinerary_agent",

]


# ============================================================
# SHARED LLM HELPERS
# ============================================================


def _llm_text(
    system_prompt: str,
    user_prompt: str
) -> str:

    response = llm.invoke(
        [

            SystemMessage(
                content=system_prompt
            ),

            HumanMessage(
                content=user_prompt
            ),

        ]
    )

    return str(
        response.content
    )


def _json_from_llm(
    text: str
) -> dict[str, Any]:

    """
    Extract the first complete JSON object
    returned by the model.
    """

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start == -1
        or end == -1
        or end < start
    ):

        raise ValueError(
            "The model did not return a JSON object."
        )

    return json.loads(
        text[
            start:end + 1
        ]
    )


def _empty_constraints() -> dict[str, Any]:

    return {

        "destination": "",

        "origin": "",

        "duration": "",

        "travel_date": "",

        "budget": "",

        "travel_style": "",

        "special_preferences": [],

    }


# ============================================================
# TOKEN / PROMPT SIZE CONTROL
# ============================================================


def _compact_text(
    text: Any,
    max_chars: int = 2500
) -> str:

    """
    Limit large MCP or agent outputs before
    sending them into another LLM prompt.

    This prevents large external tool responses
    from causing Groq token-limit errors.
    """

    if text is None:

        return ""

    text = str(
        text
    ).strip()

    if len(text) <= max_chars:

        return text

    return (

        text[:max_chars]

        + "\n\n"

        "[Additional content omitted to control prompt size.]"

    )


# ============================================================
# SUPERVISOR AGENT + INPUT GUARDRAIL
# ============================================================


def supervisor_agent(
    state: TravelState
):

    query = state[
        "user_query"
    ]

    llm_calls = state.get(
        "llm_calls",
        0
    )

    # --------------------------------------------------------
    # INPUT GUARDRAIL
    # --------------------------------------------------------

    guardrail_prompt = f"""
Determine whether the following request belongs to travel
planning or travel information.

Valid requests can include:

- destinations
- flights
- hotels
- weather
- budgets
- visas
- transportation
- sightseeing
- food
- packing
- itineraries

Block clearly unrelated requests and requests asking for
harmful or illegal instructions.

A request can be a valid travel request even if some
details are missing.

However, if the request is too vague to identify ANY
destination or travel-related objective, it should be
allowed through the guardrail but handled as an
incomplete travel request.

Return strict JSON only:

{{
  "allowed": true,
  "reason": ""
}}

User request:

{query}
"""

    # Fail open if the guardrail itself has
    # a temporary parsing/model problem.

    try:

        guardrail_raw = _llm_text(

            "You are the input guardrail for a "
            "travel-planning application. "
            "Return strict JSON only.",

            guardrail_prompt,

        )

        guardrail_result = _json_from_llm(
            guardrail_raw
        )

        allowed = bool(
            guardrail_result.get(
                "allowed",
                True
            )
        )

        guardrail_reason = str(
            guardrail_result.get(
                "reason",
                ""
            )
        ).strip()

        llm_calls += 1

    except Exception as exc:

        print(
            f"Guardrail fallback used: {exc}"
        )

        allowed = True

        guardrail_reason = (
            "Guardrail validation fallback "
            "allowed the request."
        )

    # --------------------------------------------------------
    # BLOCK INVALID REQUEST
    # --------------------------------------------------------

    if not allowed:

        reason = (

            guardrail_reason

            or

            "TripMate AI can only help with "
            "travel-planning requests. "
            "Please ask about a destination, "
            "flight, hotel, weather, budget, "
            "or itinerary."

        )

        return {

            "guardrail_allowed":
                False,

            "guardrail_reason":
                reason,

            "selected_agents":
                [],

            "trip_constraints":
                _empty_constraints(),

            "supervisor_reasoning":
                reason,

            "final_response":
                reason,

            "messages": [

                AIMessage(

                    content=(

                        "Guardrail blocked request: "

                        f"{reason}"

                    )

                )

            ],

            "llm_calls":
                llm_calls,

        }

    # --------------------------------------------------------
    # SUPERVISOR ROUTING
    # --------------------------------------------------------

    supervisor_prompt = f"""
You are the supervisor of a multi-agent
travel-planning system.

Choose only the specialist agents needed
for the user's request.

Available agents:

- flight_agent:
  flights, airports, airlines, routes,
  airfare, or booking advice

- hotel_agent:
  hotels, accommodation, neighborhoods,
  or places to stay

- weather_agent:
  weather, climate, season, forecast,
  or packing advice

- budget_agent:
  cost, affordability, price limits,
  or budget feasibility

- itinerary_agent:
  creates the integrated travel plan
  and must always be included.

IMPORTANT INCOMPLETE-REQUEST RULE:

First extract the travel constraints.

If the request does NOT contain a usable
destination, DO NOT call any specialist
travel agents.

For example:

"I want to travel somewhere."

must NOT trigger:

- flight_agent
- hotel_agent
- weather_agent
- budget_agent
- itinerary_agent

Instead, return an empty selected_agents
list and explain that a destination is
required before a travel plan can be created.

Do NOT invent or guess a destination.

If a destination IS provided, select the
appropriate specialist agents and always
include itinerary_agent.

Return strict JSON only using this schema:

{{
  "selected_agents": [],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_date": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:

{query}
"""

    try:

        supervisor_raw = _llm_text(

            "You route work to travel specialist "
            "agents. Return strict JSON only.",

            supervisor_prompt,

        )

        parsed = _json_from_llm(
            supervisor_raw
        )

        requested_agents = parsed.get(
            "selected_agents",
            []
        )

        selected_agents = [

            name

            for name in AGENT_ORDER

            if name in requested_agents

            and name in KNOWN_AGENTS

        ]

        constraints = (
            _empty_constraints()
        )

        parsed_constraints = parsed.get(
            "trip_constraints",
            {}
        )

        if isinstance(
            parsed_constraints,
            dict
        ):

            constraints.update(
                parsed_constraints
            )

        destination = str(
            constraints.get(
                "destination",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # MISSING DESTINATION
        # ----------------------------------------------------

        if not destination:

            selected_agents = []

            reasoning = (
                "A destination is required before "
                "TripMate AI can create a travel plan. "
                "Please provide a destination."
            )

            return {

                "guardrail_allowed":
                    True,

                "guardrail_reason":
                    guardrail_reason,

                "selected_agents":
                    [],

                "trip_constraints":
                    constraints,

                "supervisor_reasoning":
                    reasoning,

                "final_response":
                    reasoning,

                "messages": [

                    AIMessage(

                        content=reasoning

                    )

                ],

                "llm_calls":
                    llm_calls + 1,

            }

        # Itinerary must always be included
        # for a valid destination-based request.

        if (
            "itinerary_agent"
            not in selected_agents
        ):

            selected_agents.append(
                "itinerary_agent"
            )

        reasoning = str(
            parsed.get(
                "reasoning",
                ""
            )
        ).strip()

        llm_calls += 1

    except Exception as exc:

        print(
            f"Supervisor fallback used: {exc}"
        )

        # Preserve original full workflow
        # if supervisor parsing fails.

        selected_agents = (
            AGENT_ORDER.copy()
        )

        constraints = (
            _empty_constraints()
        )

        reasoning = (

            "Supervisor parsing failed, so "

            "the original full travel workflow "

            "was selected as a safe fallback."

        )

    return {

        "guardrail_allowed":
            True,

        "guardrail_reason":
            guardrail_reason,

        "selected_agents":
            selected_agents,

        "trip_constraints":
            constraints,

        "supervisor_reasoning":
            reasoning,

        "messages": [

            AIMessage(

                content=(

                    "Supervisor created "

                    "the agent plan."

                )

            )

        ],

        "llm_calls":
            llm_calls,

    }


# ============================================================
# GUARDRAIL BLOCKED RESPONSE
# ============================================================


def guardrail_blocked_agent(
    state: TravelState
):

    reason = (

        state.get(
            "final_response"
        )

        or

        state.get(
            "guardrail_reason"
        )

        or

        "This request was blocked by "
        "the travel input guardrail."

    )

    return {

        "final_response":
            reason,

        "messages": [

            AIMessage(
                content=reason
            )

        ],

    }

# ============================================================
# INCOMPLETE TRAVEL REQUEST
# ============================================================

def incomplete_request_agent(
    state: TravelState
):
    reason = (
        "A destination is required before "
        "TripMate AI can create a travel plan. "
        "Please provide a destination."
    )

    return {
        "final_response": reason,
        "messages": [
            AIMessage(
                content=reason
            )
        ],
    }


# ============================================================
# FLIGHT AGENT
# ============================================================

from datetime import datetime, timedelta


# ------------------------------------------------------------
# CITY -> AIRPORT IATA CODE
# ------------------------------------------------------------

AIRPORT_MAP = {
    "dhaka": "DAC",
    "dubai": "DXB",
    "delhi": "DEL",
    "new delhi": "DEL",
    "mumbai": "BOM",
    "kolkata": "CCU",
    "chennai": "MAA",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "hyderabad": "HYD",
    "doha": "DOH",
    "london": "LHR",
    "paris": "CDG",
    "singapore": "SIN",
    "bangkok": "BKK",
    "istanbul": "IST",
    "new york": "JFK",
    "toronto": "YYZ",
    "sydney": "SYD",
}


# ------------------------------------------------------------
# AIRLINE -> IATA CODE
# ------------------------------------------------------------

AIRLINE_CODES = {
    "emirates": "EK",
    "biman bangladesh": "BG",
    "biman": "BG",
    "qatar airways": "QR",
    "qatar": "QR",
    "turkish airlines": "TK",
    "turkish": "TK",
    "singapore airlines": "SQ",
    "singapore": "SQ",
    "air india": "AI",
    "indigo airlines": "6E",
    "indigo": "6E",
    "etihad airways": "EY",
    "etihad": "EY",
    "british airways": "BA",
    "lufthansa": "LH",
    "klm": "KL",
    "air france": "AF",
    "qantas": "QF",
    "cathay pacific": "CX",
    "cathay": "CX",
}


# ------------------------------------------------------------
# NORMALIZE AVIATIONSTACK FLIGHT DATA
# ------------------------------------------------------------

def _normalize_flight_schedule_data(raw_data):
    """
    Convert AviationStack MCP output into clean flight records.

    Also fixes overnight flights where the API returns an
    arrival time earlier than the departure time on the same date.
    """

    try:

        # MCP normally returns:
        # [
        #     {
        #         "type": "text",
        #         "text": "[...]"
        #     }
        # ]

        if isinstance(raw_data, list):

            text_parts = []

            for item in raw_data:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        text_parts.append(
                            item.get("text", "")
                        )

            if text_parts:

                combined_text = "".join(text_parts)

                try:
                    flight_records = json.loads(
                        combined_text
                    )

                except json.JSONDecodeError:
                    flight_records = raw_data

            else:
                flight_records = raw_data

        elif isinstance(raw_data, str):

            try:
                flight_records = json.loads(raw_data)

            except json.JSONDecodeError:
                return raw_data

        else:
            flight_records = raw_data

        if not isinstance(flight_records, list):
            return flight_records

        normalized = []

        for flight in flight_records:

            if not isinstance(flight, dict):
                continue

            cleaned_flight = dict(flight)

            departure = cleaned_flight.get(
                "departure_scheduled_time"
            )

            arrival = cleaned_flight.get(
                "arrival_scheduled_time"
            )

            if departure and arrival:

                try:

                    departure_dt = datetime.fromisoformat(
                        departure
                    )

                    arrival_dt = datetime.fromisoformat(
                        arrival
                    )

                    # AviationStack can return an overnight
                    # arrival with the same calendar date.
                    #
                    # Example:
                    #
                    # Departure: 2026-10-29 22:00
                    # Arrival:   2026-10-29 01:05
                    #
                    # Correct interpretation:
                    #
                    # Arrival:   2026-10-30 01:05

                    if arrival_dt <= departure_dt:

                        arrival_dt = arrival_dt + timedelta(
                            days=1
                        )

                        cleaned_flight[
                            "arrival_scheduled_time"
                        ] = arrival_dt.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                        cleaned_flight[
                            "overnight_flight"
                        ] = True

                    else:

                        cleaned_flight[
                            "overnight_flight"
                        ] = False

                except (ValueError, TypeError):

                    cleaned_flight[
                        "overnight_flight"
                    ] = False

            normalized.append(
                cleaned_flight
            )

        return normalized

    except Exception:

        return raw_data


# ------------------------------------------------------------
# FLIGHT AGENT PROMPT
# ------------------------------------------------------------

FLIGHT_AGENT_PROMPT = """
You are the flight specialist in a multi-agent travel
planning system.

Your job is to summarize ONLY the flight information
returned by the AviationStack MCP tool.

User request:
{query}

Origin:
{origin}

Destination:
{destination}

Requested airline:
{airline}

Requested travel date:
{travel_date}

AviationStack schedule data:
{flight_data}

IMPORTANT RULES:

1. Use only the flight data provided above.

2. NEVER invent a flight number.

3. NEVER invent an airline.

4. NEVER invent departure times.

5. NEVER invent arrival times.

6. NEVER invent aircraft information.

7. NEVER invent ticket prices.

8. AviationStack schedule data does not provide ticket
   fares.

9. If fare information is unavailable, explicitly say that
   live airfare was not provided by the connected flight data.

10. If schedule data is unavailable or contains an error,
    clearly state that live flight schedules could not be
    retrieved.

11. Do not turn estimates into facts.

12. If a flight is marked as an overnight flight, make sure
    the arrival date is shown exactly as provided in the
    normalized data.

13. Only mention the destination if the returned flight data
    contains the matching destination airport.

14. Keep the response concise and useful.

Return the result using this structure:

Flight options

- Airline
- Flight number
- Departure
- Arrival
- Aircraft

Data limitations

- Mention if airfare, seat availability, or another requested
  field is unavailable.

Booking guidance

- Give general booking advice without inventing prices,
  schedules, or availability.
"""


# ------------------------------------------------------------
# FLIGHT AGENT
# ------------------------------------------------------------

def flight_agent(
    state: TravelState
):

    print(
        "\nINSIDE FLIGHT AGENT\n"
    )

    query = state["user_query"]

    constraints = state.get(
        "trip_constraints",
        {}
    )

    origin = constraints.get(
        "origin",
        ""
    )

    destination = constraints.get(
        "destination",
        ""
    )

    travel_date = constraints.get(
        "travel_date",
        ""
    )

    try:

        # ----------------------------------------------------
        # FIND AIRLINE FROM USER QUERY
        # ----------------------------------------------------

        airline_name = ""

        query_lower = query.lower()

        # Sort by length so that names such as
        # "qatar airways" are checked before "qatar".

        for airline in sorted(
            AIRLINE_CODES.keys(),
            key=len,
            reverse=True
        ):

            if airline in query_lower:

                airline_name = airline
                break

        airline_code = AIRLINE_CODES.get(
            airline_name
        )

        # ----------------------------------------------------
        # FIND AIRPORT CODES
        # ----------------------------------------------------

        departure_iata = AIRPORT_MAP.get(
            origin.lower().strip()
        )

        arrival_iata = AIRPORT_MAP.get(
            destination.lower().strip()
        )

        # ----------------------------------------------------
        # VALIDATE REQUIRED INFORMATION
        # ----------------------------------------------------

        if not airline_code:

            flight_data = (
                "Live flight schedule lookup could not be "
                "performed because the requested airline "
                "could not be mapped to a supported airline "
                "IATA code."
            )

        elif not travel_date:

            flight_data = (
                "Live flight schedule lookup requires a "
                "specific travel date."
            )

        elif not departure_iata:

            flight_data = (
                "Live flight schedule lookup could not be "
                "performed because the origin airport could "
                "not be mapped to a supported IATA code."
            )

        else:

            # ------------------------------------------------
            # CALL AVIATIONSTACK MCP
            # ------------------------------------------------

            raw_flights = asyncio.run(
                aviation_mcp_call(
                    "future_flights_arrival_departure_schedule",
                    {
                        "airport_iata_code": departure_iata,
                        "schedule_type": "departure",
                        "airline_iata": airline_code,
                        "date": travel_date,
                        "number_of_flights": 10,
                    },
                )
            )

            # ------------------------------------------------
            # NORMALIZE RAW FLIGHT DATA
            # ------------------------------------------------

            normalized_flights = (
                _normalize_flight_schedule_data(
                    raw_flights
                )
            )

            # ------------------------------------------------
            # FILTER BY DESTINATION
            # ------------------------------------------------

            if isinstance(
                normalized_flights,
                list
            ) and arrival_iata:

                destination_matches = []

                for flight in normalized_flights:

                    flight_destination = str(
                        flight.get(
                            "arrival_airport_code",
                            ""
                        )
                    ).upper()

                    if flight_destination == arrival_iata:

                        destination_matches.append(
                            flight
                        )

                # If destination information exists and
                # matches, use only those flights.
                #
                # If no matching flights are found, we do NOT
                # pretend that the returned flights are for
                # the requested destination.

                if destination_matches:

                    normalized_flights = (
                        destination_matches
                    )

                elif normalized_flights:

                    normalized_flights = []

            # ------------------------------------------------
            # NO MATCHING FLIGHTS
            # ------------------------------------------------

            if isinstance(
                normalized_flights,
                list
            ) and not normalized_flights:

                flight_data = (
                    "No matching flight schedule was found "
                    "for the requested airline, origin, "
                    "destination, and travel date."
                )

            else:

                flight_data = normalized_flights

        # ----------------------------------------------------
        # SEND ONLY GROUNDED DATA TO LLM
        # ----------------------------------------------------

        prompt = FLIGHT_AGENT_PROMPT.format(

            query=query,

            origin=origin,

            destination=destination,

            airline=(
                airline_name.title()
                if airline_name
                else "Not specified"
            ),

            travel_date=(
                travel_date
                if travel_date
                else "Not specified"
            ),

            flight_data=_compact_text(
                flight_data,
                3000
            ),
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an expert travel flight "
                        "planner. Never invent flight data."
                    )
                ),

                HumanMessage(
                    content=prompt
                ),
            ]
        )

        flight_result = response.content

    except Exception as exc:

        flight_result = (
            "Flight information unavailable: "
            f"{exc}"
        )

    return {

        "flight_results": flight_result,

        "messages": [
            AIMessage(
                content=(
                    "Flight recommendations "
                    "generated"
                )
            )
        ],

        "llm_calls": (
            state.get(
                "llm_calls",
                0
            ) + 1
        ),
    }


# ============================================================
# HOTEL AGENT
# ============================================================


def hotel_agent(
    state: TravelState
):

    query = (

        "Best hotels for "

        f"{state['user_query']}"

    )

    try:

        hotel_results = asyncio.run(

            tavily_mcp_search(
                query
            )

        )

    except Exception as exc:

        print(

            "HOTEL AGENT MCP ERROR: "

            f"{type(exc).__name__}: {exc}",

            flush=True,

        )

        hotel_results = (

            "Live hotel search is temporarily unavailable. "
            "No verified hotel information is available "
            "from the connected travel data."

        )

    return {

        "hotel_results":
            hotel_results,

        "messages": [

            AIMessage(

                content=(
                    "Hotel information processed."
                )

            )

        ],

        "llm_calls": (

            state.get(
                "llm_calls",
                0
            ) + 1

        ),

    }


# ============================================================
# WEATHER AGENT
# ============================================================


def weather_agent(
    state: TravelState
):

    city = extract_destination(
        state["user_query"]
    )

    try:

        weather_data = asyncio.run(

            weather_mcp_search(
                city
            )

        )

        forecast_data = asyncio.run(

            forecast_mcp_search(
                city
            )

        )

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as exc:

        print(

            "WEATHER AGENT MCP ERROR: "

            f"{type(exc).__name__}: {exc}",

            flush=True,

        )

        weather_results = (

            f"Live weather information for "
            f"{city} is temporarily unavailable. "

            "Give general seasonal guidance and "
            "advise the traveler to verify the "
            "forecast before departure."

        )

    return {

        "weather_results":
            weather_results,

        "messages": [

            AIMessage(

                content=(
                    "Weather information processed."
                )

            )

        ],

    }


# ============================================================
# BUDGET AGENT
# ============================================================


def budget_agent(state: TravelState):
    print("INSIDE BUDGET AGENT")

    prompt = f"""
Create a travel budget using ONLY information explicitly contained
in the specialist-agent results below.

User Query:
{state['messages'][0].content}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{_compact_text(state.get('flight_results', ''), 1800)}

Hotel Results:
{_compact_text(state.get('hotel_results', ''), 1400)}

Weather Results:
{_compact_text(state.get('weather_results', ''), 1000)}

STRICT SOURCE-GROUNDING RULES:

1. Flight Results are the ONLY authoritative source for airfare.

2. NEVER invent, estimate, approximate, or infer airfare.

3. NEVER use general knowledge to create flight prices.

4. NEVER create a return-flight price unless an actual return
   airfare is explicitly present in Flight Results.

5. If Flight Results contain no actual airfare, write exactly:

   "Live airfare was not provided by the connected flight data."

6. Hotel Results are the ONLY authoritative source for hotel pricing.

7. If Hotel Results do not contain an actual hotel price, DO NOT
   create or estimate a hotel price.

8. Food, transportation, activities, attractions, visa fees,
   insurance, or miscellaneous costs MUST NOT be assigned numerical
   prices unless an actual numerical price is explicitly present in
   the provided specialist-agent results.

9. NEVER use typical prices, approximate prices, market averages,
   destination knowledge, previous experience, or assumptions to
   create numerical costs.

10. NEVER create a numerical daily budget.

11. NEVER create a numerical total when one or more required cost
    categories are unavailable.

12. Do not convert qualitative information into numerical estimates.

13. If a cost category has no verified numerical price in the
    specialist-agent results, write:

    "No verified price was provided by the connected travel data."

14. Clearly distinguish between verified prices and unavailable
    prices.

15. Do not add visa fees, health costs, insurance costs, baggage
    costs, taxes, or other charges unless an actual price is present
    in the provided data.

16. The budget must not contain information obtained from general
    world knowledge.

BUDGET FORMAT:

Flight:
- Use the exact airfare information from Flight Results if present.
- Otherwise write:
  "Live airfare was not provided by the connected flight data."

Hotel:
- Use only an actual price explicitly present in Hotel Results.
- Otherwise write:
  "No verified price was provided by the connected travel data."

Local Transportation:
- Use only a numerical price explicitly provided by the specialist
  data.
- Otherwise write:
  "No verified price was provided by the connected travel data."

Food:
- Use only a numerical price explicitly provided by the specialist
  data.
- Otherwise write:
  "No verified price was provided by the connected travel data."

Activities / Attractions:
- Use only numerical prices explicitly provided by the specialist
  data.
- Otherwise write:
  "No verified price was provided by the connected travel data."

Miscellaneous:
- Use only numerical prices explicitly provided by the specialist
  data.
- Otherwise write:
  "No verified price was provided by the connected travel data."

Estimated Total:
- Calculate a total ONLY when all included numerical costs are
  explicitly supported by the specialist-agent results.
- If airfare or other required costs are unavailable, do NOT invent
  values and do NOT calculate a misleading total.
- Instead write:
  "A verified total cannot be calculated because some required
   prices were not provided by the connected travel data."

Return only the budget analysis.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strictly source-grounded travel budget "
                    "analyst. Use ONLY numerical prices explicitly "
                    "provided in the specialist-agent results. "
                    "Never invent, estimate, approximate, or infer "
                    "prices from general knowledge. "
                    "Never invent airfare or return-flight costs. "
                    "If airfare is unavailable, state exactly: "
                    "'Live airfare was not provided by the connected "
                    "flight data.' "
                    "If another cost has no verified price, state: "
                    "'No verified price was provided by the connected "
                    "travel data.' "
                    "Never calculate a total when required prices "
                    "are unavailable."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "current_agent": "budget_agent",
    }

# ============================================================
# ITINERARY AGENT
# ============================================================


def itinerary_agent(
    state: TravelState
):

    prompt = f"""
Create a complete travel itinerary for the user.

============================================================
USER REQUEST
============================================================

{state['user_query']}


============================================================
TRIP CONSTRAINTS
============================================================

{state.get('trip_constraints', {})}


============================================================
FLIGHT RESULTS — AUTHORITATIVE FLIGHT SOURCE
============================================================

{_compact_text(
    state.get(
        'flight_results',
        ''
    ),
    1800
)}


============================================================
HOTEL RESULTS — AUTHORITATIVE HOTEL SOURCE
============================================================

{_compact_text(
    state.get(
        'hotel_results',
        ''
    ),
    1400
)}


============================================================
WEATHER RESULTS — AUTHORITATIVE WEATHER SOURCE
============================================================

{_compact_text(
    state.get(
        'weather_results',
        ''
    ),
    1000
)}


============================================================
BUDGET RESULTS — AUTHORITATIVE BUDGET SOURCE
============================================================

{_compact_text(
    state.get(
        'budget_results',
        ''
    ),
    1400
)}


============================================================
STRICT GROUNDING RULES
============================================================

The specialist-agent results above are the ONLY factual
sources available to you.

You MUST NOT use your own general knowledge to fill missing
travel information.

You MUST NOT invent, assume, infer, or present unsupported
travel facts as facts.

If information is not present in the specialist-agent results,
omit it or clearly state:

"Information was not available from the connected travel data."


============================================================
FLIGHT RULES
============================================================

Flight Results are the ONLY authoritative source for flights.

- Use ONLY flights explicitly provided by Flight Agent.
- NEVER invent a flight number.
- NEVER invent an airline.
- NEVER add alternative airlines.
- NEVER change a departure date.
- NEVER change a departure time.
- NEVER change an arrival date.
- NEVER change an arrival time.
- NEVER change an aircraft type.
- NEVER change a terminal.
- NEVER invent a flight duration.
- NEVER invent ticket prices.
- NEVER create a fare range.
- NEVER create a typical airfare estimate.
- NEVER invent seat availability.
- NEVER invent booking availability.
- NEVER claim that a flight is booked or confirmed.
- NEVER create a return flight unless return-flight data is
  explicitly present in Flight Results.
- NEVER turn an outbound flight into a return flight.

If Flight Results contain an overnight flight, preserve its
exact next-day arrival date.

If Flight Results do not contain airfare, write exactly:

"Live airfare was not provided by the connected flight data."


============================================================
HOTEL RULES
============================================================

Hotel Results are the ONLY authoritative source for hotel
information.

- Use ONLY hotels explicitly provided by Hotel Agent.
- Do NOT invent hotel names.
- Do NOT invent hotel amenities.
- Do NOT invent hotel ratings.
- Do NOT invent hotel prices.
- Do NOT invent hotel availability.
- Do NOT invent hotel locations or neighborhood claims.
- Do NOT claim that breakfast, Wi-Fi, pools, gyms, restaurants,
  or other amenities are available unless explicitly stated in
  Hotel Results.

If Hotel Results do not contain usable hotel information, say:

"Information was not available from the connected travel data."


============================================================
WEATHER RULES
============================================================

Weather Results are the ONLY authoritative source for weather.

- Use ONLY weather information provided by Weather Agent.
- Do NOT generate seasonal weather information from general
  knowledge.
- Do NOT predict weather outside the supplied forecast period.
- Do NOT describe October weather unless Weather Results
  explicitly contain October data.
- Do NOT present current weather as the forecast for the
  user's future travel dates.
- Clearly state when the requested travel-date forecast is
  unavailable.


============================================================
ACTIVITY RULES
============================================================

Activities may be included as itinerary suggestions.

However:

- Do NOT present attraction prices unless provided by a
  specialist agent.
- Do NOT present attraction opening hours unless provided by
  a specialist agent.
- Do NOT claim attraction availability.
- Do NOT claim booking availability.
- Do NOT claim that tickets are available.
- Do NOT claim that an activity is pre-booked.
- Do NOT invent transportation prices.
- Do NOT invent restaurant prices.
- Do NOT invent restaurant availability.

When information is unavailable, keep the activity as a
general planning suggestion without unsupported factual
details.

For example, write:

"Consider visiting a major Dubai attraction."

NOT:

"Visit Burj Khalifa at 10 AM and purchase a $45 ticket."


============================================================
VISA / HEALTH / ENTRY RULES
============================================================

Do NOT provide visa, immigration, entry, vaccination, health,
or travel-restriction information unless it is explicitly
present in the specialist-agent results.

Do NOT write statements such as:

- "You need a UAE visa."
- "Visa on arrival is available."
- "Apply for an e-visa."
- "Check COVID requirements."
- "Vaccination is required."

If such information is not available, omit it.


============================================================
BUDGET RULES
============================================================

Use Budget Results as the source for budget figures.

- Keep all non-flight costs clearly labeled as estimates.
- Do NOT invent airfare.
- Do NOT create round-trip airfare.
- Do NOT create a return-flight cost.
- Do NOT introduce numerical flight prices from general
  knowledge.
- Do NOT repeat an unsupported flight price from Budget Results.
- Flight pricing must come exclusively from Flight Results.

If Flight Results contain no airfare, state exactly:

"Live airfare was not provided by the connected flight data."

The estimated total MUST exclude airfare when airfare is
unavailable.


============================================================
RECOMMENDATION RULES
============================================================

Final recommendations must be based only on information
already provided by the specialist agents.

Do NOT use recommendations as a way to introduce unsupported
facts.

For example:

Allowed:
"Verify current flight schedules before booking."

Not allowed:
"Book Emirates for the best baggage allowance."

Allowed:
"Check current hotel listings that match your mid-range
preference."

Not allowed:
"Choose a 4-star hotel with free Wi-Fi and breakfast."

Allowed:
"Check the weather forecast closer to the travel date."

Not allowed:
"October will be warm and sunny."


============================================================
ITINERARY FORMAT
============================================================

Create a practical draft itinerary suitable for human review.

Use these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Keep the draft concise and easy to follow.

The draft must summarize available specialist-agent data
rather than create new factual information.


============================================================
FINAL SAFETY CHECK
============================================================

Before producing the answer, check the entire itinerary.

Remove any statement that introduces information not contained
in the specialist-agent results.

In particular, remove:

- unsupported airline names
- unsupported flight details
- unsupported airfare
- unsupported hotel amenities
- unsupported hotel prices
- unsupported visa claims
- unsupported health claims
- unsupported baggage claims
- unsupported attraction prices
- unsupported attraction hours
- unsupported booking claims
- unsupported availability claims
- unsupported seasonal weather claims
- unsupported travel restrictions

When in doubt, OMIT the claim.


============================================================
END OF INSTRUCTIONS
============================================================
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert AI travel itinerary planner. "
                    "Use ONLY the information provided by the "
                    "specialist agents. Do not use general world "
                    "knowledge to fill missing information. "
                    "Never invent flights, airlines, flight dates, "
                    "times, aircraft, terminals, airfare, hotel "
                    "amenities, hotel prices, visa requirements, "
                    "health requirements, baggage policies, "
                    "attraction prices, opening hours, booking "
                    "availability, or seasonal weather information. "
                    "If information is unavailable, omit it or state: "
                    "'Information was not available from the connected "
                    "travel data.' "
                    "If airfare is unavailable, state exactly: "
                    "'Live airfare was not provided by the connected "
                    "flight data.'"
                )
            ),
            HumanMessage(
                content=prompt
            ),
        ]
    )

    approval_request = (
        "Please review the generated draft "
        "itinerary. Approve it to create the "
        "final polished plan, or provide "
        "feedback for revision."
    )

    return {
        "itinerary": response.content,

        "approval_request": approval_request,

        "messages": [
            AIMessage(
                content=(
                    "Draft itinerary created "
                    "for human review."
                )
            )
        ],

        "llm_calls": (
            state.get(
                "llm_calls",
                0
            ) + 1
        ),
    }
# ============================================================
# HUMAN-IN-THE-LOOP APPROVAL
# ============================================================


def human_approval_agent(
    state: TravelState
):

    # Do not wrap interrupt() in try/except.
    # LangGraph uses interrupt() to pause execution.

    review = interrupt(
        {
            "question": (
                "Do you approve this itinerary?"
            ),

            "draft_itinerary": (
                state.get(
                    "itinerary",
                    ""
                )
            ),

            "approval_request": (
                state.get(
                    "approval_request",
                    ""
                )
            ),

            "selected_agents": (
                state.get(
                    "selected_agents",
                    []
                )
            ),

            "supervisor_reasoning": (
                state.get(
                    "supervisor_reasoning",
                    ""
                )
            ),

            "expected_response": {
                "approved": True,
                "feedback": (
                    "Optional revision feedback"
                ),
            },
        }
    )

    approved = bool(
        review.get(
            "approved",
            False
        )
    )

    human_feedback = str(
        review.get(
            "feedback",
            ""
        )
    ).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [
            AIMessage(
                content=(
                    "Human approval step completed."
                )
            )
        ],
    }


# ============================================================
# FINAL RESPONSE AGENT
# ============================================================


def final_agent(
    state: TravelState
):

    if state.get(
        "approved",
        False
    ):

        review_instruction = (
            "The user approved the draft. "
            "Preserve its decisions while "
            "polishing the presentation."
        )

    else:

        review_instruction = f"""
The user requested a revision.

Apply this feedback carefully:

{_compact_text(
    state.get(
        "human_feedback",
        ""
    ),
    1000
)}

Revise only what is necessary to
address the user's feedback.
"""

    final_prompt = f"""
Generate the final travel response for the user.

============================================================
HUMAN REVIEW
============================================================

{review_instruction}


============================================================
USER REQUEST
============================================================

{state['user_query']}


============================================================
SUPERVISOR CONSTRAINTS
============================================================

{state.get('trip_constraints', {})}


============================================================
FLIGHT DATA — AUTHORITATIVE SOURCE
============================================================

{_compact_text(
    state.get(
        'flight_results',
        ''
    ),
    1800
)}


============================================================
HOTEL DATA
============================================================

{_compact_text(
    state.get(
        'hotel_results',
        ''
    ),
    1600
)}


============================================================
WEATHER DATA
============================================================

{_compact_text(
    state.get(
        'weather_results',
        ''
    ),
    1200
)}


============================================================
BUDGET DATA
============================================================

{_compact_text(
    state.get(
        'budget_results',
        ''
    ),
    1600
)}


============================================================
DRAFT ITINERARY
============================================================

{_compact_text(
    state.get(
        'itinerary',
        ''
    ),
    2800
)}


============================================================
STRICT DATA CONSISTENCY RULES
============================================================

The specialist agents above provide the source information
for this final response.

You MUST follow these rules:

1. NEVER invent flight information.

2. NEVER change a flight number.

3. NEVER change a flight's departure time.

4. NEVER change a flight's arrival time.

5. NEVER change a flight's departure date.

6. NEVER change a flight's arrival date.

7. NEVER invent ticket prices.

8. NEVER convert an unavailable fare into a specific numerical fare.

9. If the Flight Agent says that airfare is unavailable, preserve
   that limitation.

10. If the Flight Agent provides a specific flight date and time,
    reproduce that information exactly.

11. If a flight is marked as an overnight flight, preserve its
    next-day arrival date exactly.

12. Do not infer a different flight duration from clock times.

13. Do not replace real flight data with typical or estimated
    flight information.

14. Do not invent seat availability.

15. Do not invent booking availability.

16. Do not claim that a flight is booked or confirmed.

17. Do not create return-flight details unless return-flight
    information is explicitly present in Flight Results.

18. Hotel prices must remain clearly labeled as approximate
    unless live pricing was actually provided.

19. Weather information must not be presented as guaranteed
    future conditions when it is only general or seasonal guidance.

20. Budget estimates must remain estimates.

21. If two pieces of information conflict, prefer the specialist
    result that directly provides that information rather than
    inventing a value.

22. Preserve the user's requested destination, origin, and travel date.

23. Do not silently change the user's trip dates.

24. The final response should summarize available information,
    not create new facts.

25. If airfare is unavailable, explicitly state exactly:

    "Live airfare was not provided by the connected flight data."

26. Do not include any numerical airfare anywhere in the final
    response when Flight Results do not provide airfare.

27. Do not repeat an unsupported flight price from the draft
    itinerary or Budget Agent.

28. Do not create a round-trip price for a one-way request.

29. Do not describe an outbound flight as a return flight.

30. If the draft itinerary contains unsupported flight claims,
    remove those claims rather than preserving them.


============================================================
FINAL FORMAT
============================================================

Format the final answer clearly using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Keep the response practical and concise.

For every flight shown in the final response:

- Use the exact flight number from Flight Agent data.
- Use the exact departure date and time.
- Use the exact arrival date and time.
- Use the exact aircraft when available.
- Use the exact terminal when available.
- Do not add a fare unless Flight Agent actually provided one.
- If fare is unavailable, explicitly state:

  "Live airfare was not provided by the connected flight data."

If the user's request is one-way and no return-flight data exists,
do not create a return flight, return date, return fare, or return
itinerary.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a professional AI travel planner. "
                    "You must preserve factual consistency across "
                    "specialist-agent results. Flight Results are "
                    "the authoritative source for flight information. "
                    "Never invent flight data, ticket prices, fare "
                    "ranges, seat availability, booking status, or "
                    "return flights. If airfare is unavailable, state "
                    "exactly: 'Live airfare was not provided by the "
                    "connected flight data.'"
                )
            ),
            HumanMessage(
                content=final_prompt
            ),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [
            response
        ],
        "llm_calls": (
            state.get(
                "llm_calls",
                0
            ) + 1
        ),
    }


# ============================================================
# DYNAMIC SUPERVISOR ROUTING
# ============================================================


ROUTE_MAP = {

    "guardrail_blocked":
        "guardrail_blocked",

    "incomplete_request":
        "incomplete_request",

    "flight_agent":
        "flight_agent",

    "hotel_agent":
        "hotel_agent",

    "weather_agent":
        "weather_agent",

    "budget_agent":
        "budget_agent",

    "itinerary_agent":
        "itinerary_agent",

}


def _selected_agents(
    state: TravelState
) -> list[str]:

    selected = state.get(
        "selected_agents",
        []
    )

    return [

        agent

        for agent in AGENT_ORDER

        if agent in selected

    ]


def route_from_supervisor(
    state: TravelState
) -> str:

    if not state.get(
        "guardrail_allowed",
        True
    ):

        return "guardrail_blocked"

    selected = _selected_agents(
        state
    )

    # No destination / incomplete travel request.
    # The supervisor intentionally selected no agents.
    # Route directly to the end instead of running
    # itinerary generation or Human-in-the-Loop approval.

    if not selected:

        return "incomplete_request"

    return selected[0]


def route_after_agent(
    current_agent: str
):

    def route(
        state: TravelState
    ) -> str:

        selected = _selected_agents(
            state
        )

        current_index = (
            AGENT_ORDER.index(
                current_agent
            )
        )

        for next_agent in (
            AGENT_ORDER[
                current_index + 1:
            ]
        ):

            if next_agent in selected:

                return next_agent

        return "itinerary_agent"

    return route


# ============================================================
# BUILD LANGGRAPH
# ============================================================


graph = StateGraph(
    TravelState
)


graph.add_node(
    "supervisor",
    supervisor_agent
)


graph.add_node(
    "guardrail_blocked",
    guardrail_blocked_agent
)

graph.add_node(
    "incomplete_request",
    incomplete_request_agent
)


graph.add_node(
    "flight_agent",
    flight_agent
)


graph.add_node(
    "hotel_agent",
    hotel_agent
)


graph.add_node(
    "weather_agent",
    weather_agent
)


graph.add_node(
    "budget_agent",
    budget_agent
)


graph.add_node(
    "itinerary_agent",
    itinerary_agent
)


graph.add_node(
    "human_approval",
    human_approval_agent
)


graph.add_node(
    "final_agent",
    final_agent
)


# ------------------------------------------------------------
# GRAPH EDGES
# ------------------------------------------------------------


graph.add_edge(
    START,
    "supervisor"
)


graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    ROUTE_MAP
)


graph.add_conditional_edges(
    "flight_agent",
    route_after_agent(
        "flight_agent"
    ),
    ROUTE_MAP
)


graph.add_conditional_edges(
    "hotel_agent",
    route_after_agent(
        "hotel_agent"
    ),
    ROUTE_MAP
)


graph.add_conditional_edges(
    "weather_agent",
    route_after_agent(
        "weather_agent"
    ),
    ROUTE_MAP
)


graph.add_conditional_edges(
    "budget_agent",
    route_after_agent(
        "budget_agent"
    ),
    ROUTE_MAP
)


graph.add_edge(
    "itinerary_agent",
    "human_approval"
)


graph.add_edge(
    "human_approval",
    "final_agent"
)


graph.add_edge(
    "final_agent",
    END
)


graph.add_edge(
    "guardrail_blocked",
    END
)

graph.add_edge(
    "incomplete_request",
    END
)


# ============================================================
# POSTGRESQL CHECKPOINTER
# ============================================================


DATABASE_URL = get_database_url()


_conn = psycopg.connect(

    DATABASE_URL,

    autocommit=True,

    row_factory=dict_row,

)


checkpointer = PostgresSaver(
    _conn
)


checkpointer.setup()


travel_graph = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# FASTAPI-FACING HELPERS
# ============================================================


def _interrupt_payload(
    result: dict[str, Any]
) -> dict[str, Any] | None:

    interrupts = result.get(
        "__interrupt__",
        []
    )

    if not interrupts:

        return None

    first_interrupt = interrupts[0]

    payload = getattr(
        first_interrupt,
        "value",
        first_interrupt
    )

    return (

        payload

        if isinstance(
            payload,
            dict
        )

        else {
            "value": payload
        }

    )


def _serialize_result(
    result: dict[str, Any],
    thread_id: str
) -> dict[str, Any]:

    messages = result.get(
        "messages",
        []
    )

    last_message = (

        messages[-1].content

        if messages

        else ""

    )

    answer = (

        result.get(
            "final_response"
        )

        or last_message

    )

    interrupt_payload = (
        _interrupt_payload(
            result
        )
    )

    if interrupt_payload:

        answer = (

            interrupt_payload.get(
                "draft_itinerary"
            )

            or

            result.get(
                "itinerary",
                ""
            )

        )

    return {

        "thread_id":
            thread_id,

        "answer":
            answer,

        "requires_approval":
            interrupt_payload is not None,

        "approval_request": (

            interrupt_payload.get(
                "approval_request",
                ""
            )

            if interrupt_payload

            else result.get(
                "approval_request",
                ""
            )

        ),

        "flight_results":
            result.get(
                "flight_results",
                ""
            ),

        "hotel_results":
            result.get(
                "hotel_results",
                ""
            ),

        "weather_results":
            result.get(
                "weather_results",
                ""
            ),

        "budget_results":
            result.get(
                "budget_results",
                ""
            ),

        "itinerary": (

            interrupt_payload.get(
                "draft_itinerary",
                ""
            )

            if interrupt_payload

            else result.get(
                "itinerary",
                ""
            )

        ),

        "selected_agents":
            result.get(
                "selected_agents",
                []
            ),

        "trip_constraints":
            result.get(
                "trip_constraints",
                {}
            ),

        "supervisor_reasoning":
            result.get(
                "supervisor_reasoning",
                ""
            ),

        "guardrail_allowed":
            result.get(
                "guardrail_allowed",
                True
            ),

        "guardrail_reason":
            result.get(
                "guardrail_reason",
                ""
            ),

        "approved":
            result.get(
                "approved"
            ),

        "human_feedback":
            result.get(
                "human_feedback",
                ""
            ),

        "llm_calls":
            result.get(
                "llm_calls",
                0
            ),

    }


# ============================================================
# START TRAVEL AGENT
# ============================================================


def run_travel_agent(
    user_input: str,
    thread_id: str | None = None
):

    """
    Start a new travel-planning run
    and pause at human approval.
    """

    if not thread_id:

        thread_id = (

            f"user_{uuid.uuid4().hex}"

        )

    config = {

        "configurable": {

            "thread_id":
                thread_id

        }

    }

    result = travel_graph.invoke(

        {

            "messages": [

                HumanMessage(
                    content=user_input
                )

            ],

            "user_query":
                user_input,

            "guardrail_allowed":
                True,

            "guardrail_reason":
                "",

            "selected_agents":
                [],

            "trip_constraints":
                _empty_constraints(),

            "supervisor_reasoning":
                "",

            "flight_results":
                "",

            "hotel_results":
                "",

            "weather_results":
                "",

            "budget_results":
                "",

            "itinerary":
                "",

            "approval_request":
                "",

            "approved":
                False,

            "human_feedback":
                "",

            "final_response":
                "",

            "llm_calls":
                0,

        },

        config=config,

    )

    return _serialize_result(
        result,
        thread_id
    )


# ============================================================
# RESUME TRAVEL AGENT
# ============================================================


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):

    """
    Resume the paused LangGraph thread
    after human review.
    """

    if not thread_id:

        raise ValueError(

            "thread_id is required "
            "to resume a travel plan."

        )

    config = {

        "configurable": {

            "thread_id":
                thread_id

        }

    }

    result = travel_graph.invoke(

        Command(

            resume={

                "approved":
                    approved,

                "feedback":
                    feedback.strip(),

            }

        ),

        config=config,

    )

    return _serialize_result(
        result,
        thread_id
    )