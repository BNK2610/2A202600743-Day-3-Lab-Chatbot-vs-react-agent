import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import unicodedata


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _normalize(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


@lru_cache(maxsize=None)
def _load_csv(filename: str) -> List[Dict[str, str]]:
    path = DATA_DIR / filename
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_trip(row: Dict[str, str]) -> Dict[str, Any]:
    trip = dict(row)
    trip["price"] = _to_int(row.get("price"), 0)
    trip["seats_left"] = _to_int(row.get("seats_left"), 0)
    trip["baggage_fee"] = _to_int(row.get("baggage_fee"), 0)
    return trip


def _as_hotel(row: Dict[str, str]) -> Dict[str, Any]:
    hotel = dict(row)
    hotel["stars"] = _to_int(row.get("stars"), 0)
    hotel["price_per_night"] = _to_int(row.get("price_per_night"), 0)
    hotel["available_rooms"] = _to_int(row.get("available_rooms"), 0)
    hotel["near_center"] = _to_bool(row.get("near_center"))
    hotel["breakfast_included"] = _to_bool(row.get("breakfast_included"))
    return hotel


def _as_activity(row: Dict[str, str]) -> Dict[str, Any]:
    activity = dict(row)
    activity["price"] = _to_int(row.get("price"), 0)
    activity["duration_hours"] = _to_float(row.get("duration_hours"), 0.0)
    return activity


def resolve_city(city_or_code: str) -> Dict[str, Any]:
    """
    Resolve a user-facing city name or airport code to the canonical city record.
    """
    query = _normalize(city_or_code)
    if not query:
        return {
            "status": "MISSING_CITY",
            "message": "City or airport code is required.",
        }

    for row in _load_csv("city_aliases.csv"):
        if query in {
            _normalize(row.get("alias")),
            _normalize(row.get("code")),
            _normalize(row.get("city")),
            _normalize(row.get("airport_name")),
        }:
            return {
                "status": "OK",
                "input": city_or_code,
                "code": row["code"],
                "city": row["city"],
                "airport_name": row["airport_name"],
            }

    return {
        "status": "UNKNOWN_CITY",
        "input": city_or_code,
        "message": f"Could not resolve city or airport code: {city_or_code}",
    }


def _city_code(value: str) -> Optional[str]:
    resolved = resolve_city(value)
    if resolved.get("status") == "OK":
        return resolved["code"]
    return str(value or "").strip().upper() or None


def _city_name(value: str) -> Optional[str]:
    resolved = resolve_city(value)
    if resolved.get("status") == "OK":
        return resolved["city"]
    return str(value or "").strip() or None


def search_trips(
    origin: str,
    destination: str,
    date: str,
    max_price: Optional[int] = None,
    passengers: int = 1,
) -> Dict[str, Any]:
    """
    Search matching trips by route, date, max price, and required seats.
    """
    origin_code = _city_code(origin)
    destination_code = _city_code(destination)
    max_price_int = _to_int(max_price)
    passenger_count = max(_to_int(passengers, 1) or 1, 1)

    route_options = [
        _as_trip(row)
        for row in _load_csv("trips.csv")
        if row.get("origin") == origin_code
        and row.get("destination") == destination_code
        and row.get("date") == date
    ]
    route_options.sort(key=lambda item: item["price"])

    if not route_options:
        return {
            "status": "NO_ROUTE",
            "origin": origin_code,
            "destination": destination_code,
            "date": date,
            "results": [],
            "message": "No trips found for this route/date.",
        }

    within_budget = [
        trip
        for trip in route_options
        if max_price_int is None or trip["price"] <= max_price_int
    ]
    valid = [trip for trip in within_budget if trip["seats_left"] >= passenger_count]
    seat_limited = [
        trip for trip in within_budget if trip["seats_left"] < passenger_count
    ]

    if valid:
        return {
            "status": "OK",
            "origin": origin_code,
            "destination": destination_code,
            "date": date,
            "passengers": passenger_count,
            "max_price": max_price_int,
            "results": valid,
            "all_route_options": route_options,
        }

    if seat_limited:
        return {
            "status": "NOT_ENOUGH_SEATS",
            "origin": origin_code,
            "destination": destination_code,
            "date": date,
            "passengers": passenger_count,
            "max_price": max_price_int,
            "results": [],
            "seat_limited_options": seat_limited,
            "all_route_options": route_options,
            "message": "Matching trips exist, but they do not have enough seats.",
        }

    return {
        "status": "NO_MATCHING_TRIP",
        "origin": origin_code,
        "destination": destination_code,
        "date": date,
        "passengers": passenger_count,
        "max_price": max_price_int,
        "results": [],
        "all_route_options": route_options,
        "message": "Trips exist for this route/date, but none match the budget.",
    }


def get_weather(city: str, date: str) -> Dict[str, Any]:
    """
    Look up weather by canonical city name and date.
    """
    city_name = _city_name(city)
    for row in _load_csv("weather.csv"):
        if _normalize(row.get("city")) == _normalize(city_name) and row.get("date") == date:
            return {
                "status": "OK",
                "city": row["city"],
                "date": row["date"],
                "temp_high_c": _to_int(row.get("temp_high_c"), 0),
                "temp_low_c": _to_int(row.get("temp_low_c"), 0),
                "rain_probability": _to_float(row.get("rain_probability"), 0.0),
                "condition": row.get("condition"),
            }

    return {
        "status": "MISSING_WEATHER",
        "city": city_name,
        "date": date,
        "message": "No weather record found for this city/date.",
    }


def recommend_outfit(
    temp_high_c: int,
    rain_probability: float,
    condition: str = "",
) -> Dict[str, Any]:
    """
    Recommend clothing based on temperature and rain probability.
    """
    temp = _to_int(temp_high_c, 0) or 0
    rain = _to_float(rain_probability, 0.0) or 0.0
    normalized_condition = _normalize(condition)

    items = ["light breathable clothes"]
    if temp >= 34:
        items.extend(["hat", "sunscreen", "water bottle"])
    elif temp <= 24:
        items.append("light jacket")

    if rain >= 0.5 or "rain" in normalized_condition:
        items.extend(["foldable umbrella", "light raincoat", "quick-dry shoes"])

    return {
        "status": "OK",
        "temp_high_c": temp,
        "rain_probability": rain,
        "condition": condition,
        "recommended_items": items,
        "summary": ", ".join(items),
    }


def search_hotels(
    city: str,
    max_price_per_night: Optional[int] = None,
    rooms: int = 1,
    near_center: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Search hotels by city, nightly budget, room count, and optional center preference.
    """
    city_name = _city_name(city)
    max_price = _to_int(max_price_per_night)
    room_count = max(_to_int(rooms, 1) or 1, 1)

    candidates = [
        _as_hotel(row)
        for row in _load_csv("hotels.csv")
        if _normalize(row.get("city")) == _normalize(city_name)
    ]
    if near_center is not None:
        near_center_value = near_center if isinstance(near_center, bool) else _to_bool(near_center)
        candidates = [
            hotel for hotel in candidates if hotel["near_center"] is near_center_value
        ]

    matching = [
        hotel
        for hotel in candidates
        if (max_price is None or hotel["price_per_night"] <= max_price)
        and hotel["available_rooms"] >= room_count
    ]
    matching.sort(key=lambda item: item["price_per_night"])

    if matching:
        return {
            "status": "OK",
            "city": city_name,
            "max_price_per_night": max_price,
            "rooms": room_count,
            "results": matching,
        }

    return {
        "status": "NO_MATCHING_HOTEL",
        "city": city_name,
        "max_price_per_night": max_price,
        "rooms": room_count,
        "results": [],
        "all_city_hotels": sorted(candidates, key=lambda item: item["price_per_night"]),
        "message": "No hotels match the requested budget and room count.",
    }


def recommend_activities(
    city: str,
    condition: str = "",
    max_price: Optional[int] = None,
    activity_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Recommend activities by city, weather condition, budget, and optional type.
    """
    city_name = _city_name(city)
    max_price_int = _to_int(max_price)
    condition_norm = _normalize(condition)
    type_norm = _normalize(activity_type)

    def condition_matches(activity: Dict[str, Any]) -> bool:
        suitable = _normalize(activity.get("suitable_condition"))
        if not condition_norm:
            return True
        if "rain" in condition_norm:
            return suitable == "rainy" or _normalize(activity.get("type")) == "indoor"
        if "sun" in condition_norm or "hot" in condition_norm:
            return suitable == "sunny" or _normalize(activity.get("type")) == "outdoor"
        if "cloud" in condition_norm:
            return suitable in {"cloudy", "rainy", "sunny"}
        return suitable == condition_norm

    activities = [
        _as_activity(row)
        for row in _load_csv("activities.csv")
        if _normalize(row.get("city")) == _normalize(city_name)
    ]

    matching = []
    for activity in activities:
        if max_price_int is not None and activity["price"] > max_price_int:
            continue
        if type_norm and _normalize(activity.get("type")) != type_norm:
            continue
        if not condition_matches(activity):
            continue
        matching.append(activity)

    matching.sort(key=lambda item: (item["price"], item["duration_hours"]))

    if matching:
        return {
            "status": "OK",
            "city": city_name,
            "condition": condition,
            "max_price": max_price_int,
            "activity_type": activity_type,
            "results": matching,
        }

    return {
        "status": "NO_MATCHING_ACTIVITY",
        "city": city_name,
        "condition": condition,
        "max_price": max_price_int,
        "activity_type": activity_type,
        "results": [],
        "all_city_activities": activities,
        "message": "No activities match the requested filters.",
    }


def estimate_local_transport(city: str, transport_type: str = "taxi") -> Dict[str, Any]:
    """
    Estimate airport-to-center transport cost for a city.
    """
    city_name = _city_name(city)
    transport_norm = _normalize(transport_type or "taxi")

    for row in _load_csv("local_transport.csv"):
        if _normalize(row.get("city")) == _normalize(city_name) and _normalize(
            row.get("transport_type")
        ) == transport_norm:
            return {
                "status": "OK",
                "city": row["city"],
                "from_airport": row["from_airport"],
                "transport_type": row["transport_type"],
                "cost": _to_int(row.get("cost"), 0),
                "duration_minutes": _to_int(row.get("duration_minutes"), 0),
            }

    return {
        "status": "NO_TRANSPORT_OPTION",
        "city": city_name,
        "transport_type": transport_type,
        "message": "No local transport estimate found.",
    }


def calculate_total_cost(
    ticket_price: int = 0,
    passengers: int = 1,
    baggage_fee: int = 0,
    hotel_price_per_night: int = 0,
    nights: int = 0,
    rooms: int = 1,
    transport_cost: int = 0,
    activity_price: int = 0,
) -> Dict[str, Any]:
    """
    Calculate a simple trip total from explicit cost components.
    """
    passenger_count = max(_to_int(passengers, 1) or 1, 1)
    room_count = max(_to_int(rooms, 1) or 1, 1)
    night_count = max(_to_int(nights, 0) or 0, 0)

    ticket_total = (_to_int(ticket_price, 0) or 0) * passenger_count
    baggage_total = (_to_int(baggage_fee, 0) or 0) * passenger_count
    hotel_total = (_to_int(hotel_price_per_night, 0) or 0) * night_count * room_count
    transport_total = _to_int(transport_cost, 0) or 0
    activity_total = (_to_int(activity_price, 0) or 0) * passenger_count
    total = ticket_total + baggage_total + hotel_total + transport_total + activity_total

    return {
        "status": "OK",
        "line_items": {
            "ticket_total": ticket_total,
            "baggage_total": baggage_total,
            "hotel_total": hotel_total,
            "transport_total": transport_total,
            "activity_total": activity_total,
        },
        "total": total,
        "currency": "VND",
    }


def _tool(
    name: str,
    description: str,
    func: Callable[..., Dict[str, Any]],
    args: List[str],
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "args": args,
        "func": func,
    }


def get_travel_tools() -> List[Dict[str, Any]]:
    """
    Return the tool registry used by the ReAct agent.
    """
    return [
        _tool(
            "resolve_city",
            "Resolve a Vietnamese city name or airport code to canonical city, airport code, and airport name. Failure statuses: MISSING_CITY, UNKNOWN_CITY.",
            resolve_city,
            ["city_or_code"],
        ),
        _tool(
            "search_trips",
            "Search trips by origin, destination, date, max_price, and passengers. Returns matching trips sorted by price. Failure statuses: NO_ROUTE, NO_MATCHING_TRIP, NOT_ENOUGH_SEATS.",
            search_trips,
            ["origin", "destination", "date", "max_price", "passengers"],
        ),
        _tool(
            "get_weather",
            "Look up weather by city and date. Returns temp_high_c, temp_low_c, rain_probability, and condition. Failure status: MISSING_WEATHER.",
            get_weather,
            ["city", "date"],
        ),
        _tool(
            "recommend_outfit",
            "Recommend travel clothing from temp_high_c, rain_probability, and optional condition.",
            recommend_outfit,
            ["temp_high_c", "rain_probability", "condition"],
        ),
        _tool(
            "search_hotels",
            "Search hotels by city, max_price_per_night, rooms, and optional near_center. Failure status: NO_MATCHING_HOTEL.",
            search_hotels,
            ["city", "max_price_per_night", "rooms", "near_center"],
        ),
        _tool(
            "recommend_activities",
            "Recommend activities by city, weather condition, max_price, and optional activity_type. Failure status: NO_MATCHING_ACTIVITY.",
            recommend_activities,
            ["city", "condition", "max_price", "activity_type"],
        ),
        _tool(
            "estimate_local_transport",
            "Estimate airport-to-city-center transport cost by city and transport_type such as taxi, bus, or ride_hailing. Failure status: NO_TRANSPORT_OPTION.",
            estimate_local_transport,
            ["city", "transport_type"],
        ),
        _tool(
            "calculate_total_cost",
            "Calculate total trip cost from ticket_price, passengers, baggage_fee, hotel_price_per_night, nights, rooms, transport_cost, and activity_price.",
            calculate_total_cost,
            [
                "ticket_price",
                "passengers",
                "baggage_fee",
                "hotel_price_per_night",
                "nights",
                "rooms",
                "transport_cost",
                "activity_price",
            ],
        ),
    ]
