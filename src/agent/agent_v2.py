import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agent.agent import ReActAgent
from src.core.llm_provider import LLMProvider
from src.core.prompts import build_react_agent_v2_system_prompt
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class ReActAgentV2(ReActAgent):
    """
    Guarded ReAct agent.

    Improvements over v1:
    - accepts raw JSON tool calls as recoverable actions
    - treats clarification questions as clarification, not parser errors
    - falls back to a deterministic tool planner when the LLM skips required tools
    - prevents repeated identical tool calls
    - synthesizes a grounded final answer from observations
    """

    version = "v2"

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 8):
        super().__init__(llm=llm, tools=tools, max_steps=max_steps)
        self._city_aliases = self._load_city_aliases()

    def get_system_prompt(self) -> str:
        tool_descriptions = []
        for tool in self.tools:
            args = ", ".join(tool.get("args", []))
            tool_descriptions.append(
                f"- {tool['name']}({args}): {tool['description']}"
            )
        return build_react_agent_v2_system_prompt("\n".join(tool_descriptions))

    def run(self, user_input: str) -> str:
        logger.log_event(
            "AGENT_START",
            {
                "input": user_input,
                "model": self.llm.model_name,
                "max_steps": self.max_steps,
                "agent_version": self.version,
            },
        )

        self.history = []
        self.last_run_trace = []
        self.last_error_code = None
        self.last_steps = 0

        scratchpad = ""
        observations: List[Dict[str, Any]] = []
        called_signatures = set()

        clarification = self._required_clarification(user_input)
        if clarification:
            self.last_steps = 1
            self.last_error_code = "MISSING_REQUIRED_INFO"
            self.last_run_trace.append(
                {
                    "step": 1,
                    "type": "clarification",
                    "content": clarification,
                    "observation": {"status": "MISSING_REQUIRED_INFO"},
                }
            )
            logger.log_event(
                "AGENT_END",
                {
                    "status": "CLARIFICATION",
                    "steps": 1,
                    "error_code": self.last_error_code,
                    "agent_version": self.version,
                },
            )
            return clarification

        for step in range(1, self.max_steps + 1):
            self.last_steps = step
            prompt = self._build_prompt(user_input, scratchpad)
            result = self.llm.generate(prompt, system_prompt=self.get_system_prompt())
            content = (result.get("content") or "").strip()

            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )
            logger.log_event(
                "AGENT_LLM_RESPONSE",
                {
                    "step": step,
                    "content": content,
                    "latency_ms": result.get("latency_ms", 0),
                    "agent_version": self.version,
                },
            )

            final_answer = self._parse_final_answer(content)
            if final_answer:
                if self._has_required_observations(user_input, observations):
                    self.last_run_trace.append(
                        {"step": step, "type": "final_answer", "content": final_answer}
                    )
                    logger.log_event(
                        "AGENT_END",
                        {
                            "status": "OK",
                            "steps": step,
                            "error_code": self.last_error_code,
                            "agent_version": self.version,
                        },
                    )
                    return final_answer

            action = self._try_parse_action_v2(content)
            reason = "llm_action"

            if (
                action is None
                or self._is_duplicate_action(action, called_signatures)
                or self._should_override_llm_action(user_input, action, observations)
            ):
                fallback = self._next_guardrail_action(user_input, observations, called_signatures)
                if fallback:
                    action = fallback
                    reason = "guardrail_action"
                    logger.log_event(
                        "AGENT_V2_GUARDRAIL",
                        {"step": step, "action": action, "reason": "fallback_or_duplicate"},
                    )
                else:
                    answer = self._synthesize_final_answer(user_input, observations)
                    self.last_run_trace.append(
                        {"step": step, "type": "final_answer", "content": answer}
                    )
                    logger.log_event(
                        "AGENT_END",
                        {
                            "status": "OK",
                            "steps": step,
                            "error_code": self.last_error_code,
                            "agent_version": self.version,
                        },
                    )
                    return answer

            tool_name, tool_args = action
            signature = self._action_signature(tool_name, tool_args)
            called_signatures.add(signature)

            tool_result = self._execute_tool(tool_name, dict(tool_args))
            status = tool_result.get("status") if isinstance(tool_result, dict) else None
            if status and status != "OK":
                self.last_error_code = status

            observations.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                }
            )
            scratchpad += self._format_turn(
                f"Thought: {reason}\nAction: {json.dumps({'tool': tool_name, 'args': tool_args}, ensure_ascii=False)}",
                tool_result,
            )
            self.last_run_trace.append(
                {
                    "step": step,
                    "type": "tool_call",
                    "source": reason,
                    "tool": tool_name,
                    "args": tool_args,
                    "observation": tool_result,
                }
            )

            if self._has_required_observations(user_input, observations):
                answer = self._synthesize_final_answer(user_input, observations)
                self.last_run_trace.append(
                    {"step": step, "type": "final_answer", "content": answer}
                )
                logger.log_event(
                    "AGENT_END",
                    {
                        "status": "OK",
                        "steps": step,
                        "error_code": self.last_error_code,
                        "agent_version": self.version,
                    },
                )
                return answer

        self.last_error_code = self.last_error_code or "MAX_STEPS_EXCEEDED"
        answer = self._synthesize_final_answer(user_input, observations)
        logger.log_event(
            "AGENT_END",
            {
                "status": "MAX_STEPS_WITH_SYNTHESIS",
                "steps": self.max_steps,
                "error_code": self.last_error_code,
                "agent_version": self.version,
            },
        )
        return answer

    def _try_parse_action_v2(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        try:
            return self._parse_action(text)
        except Exception:
            pass

        payload = self._first_json_object(text)
        if payload and (payload.get("tool") or payload.get("name")):
            tool_name = payload.get("tool") or payload.get("name")
            args = payload.get("args") or payload.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(args, dict):
                return str(tool_name), args

        return None

    def _first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _is_duplicate_action(
        self,
        action: Tuple[str, Dict[str, Any]],
        called_signatures: set,
    ) -> bool:
        return self._action_signature(action[0], action[1]) in called_signatures

    def _should_override_llm_action(
        self,
        user_input: str,
        action: Tuple[str, Dict[str, Any]],
        observations: List[Dict[str, Any]],
    ) -> bool:
        tool_name, tool_args = action
        slots = self._extract_slots(user_input)
        required = self._required_tool_names(user_input, slots)
        called = {item["tool"] for item in observations}

        if tool_name == "resolve_city" and required - called:
            return True
        if required and tool_name not in required and tool_name != "calculate_total_cost":
            return True
        if tool_name == "recommend_activities":
            activity_budget = slots.get("activity_budget")
            if activity_budget is not None and self._coerce_int(tool_args.get("max_price")) != activity_budget:
                return True
            activity_type = slots.get("activity_type")
            if activity_type and tool_args.get("activity_type") != activity_type:
                return True
        return False

    def _action_signature(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        return json.dumps(
            {"tool": tool_name, "args": tool_args},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _next_guardrail_action(
        self,
        user_input: str,
        observations: List[Dict[str, Any]],
        called_signatures: set,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        slots = self._extract_slots(user_input)
        required = self._required_tool_names(user_input, slots)
        called_tools = [item["tool"] for item in observations]

        def candidate(tool_name: str, args: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
            if tool_name not in required:
                return None
            if tool_name in called_tools and tool_name not in {"get_weather"}:
                return None
            action = (tool_name, args)
            if self._is_duplicate_action(action, called_signatures):
                return None
            return action

        destination_city = slots.get("destination_city") or slots.get("city")
        route_ready = slots.get("origin") and slots.get("destination") and slots.get("date")

        if route_ready:
            action = candidate(
                "search_trips",
                {
                    "origin": slots["origin"],
                    "destination": slots["destination"],
                    "date": slots["date"],
                    "max_price": slots.get("flight_budget"),
                    "passengers": slots.get("passengers") or 1,
                },
            )
            if action:
                return action

        if destination_city and slots.get("date"):
            action = candidate(
                "get_weather",
                {
                    "city": destination_city,
                    "date": slots["date"],
                },
            )
            if action:
                return action

        weather = self._latest_observation(observations, "get_weather")
        if weather and weather.get("status") == "OK":
            action = candidate(
                "recommend_outfit",
                {
                    "temp_high_c": weather.get("temp_high_c"),
                    "rain_probability": weather.get("rain_probability"),
                    "condition": weather.get("condition"),
                },
            )
            if action:
                return action

        if destination_city:
            action = candidate(
                "search_hotels",
                {
                    "city": destination_city,
                    "max_price_per_night": slots.get("hotel_budget"),
                    "rooms": slots.get("rooms") or 1,
                },
            )
            if action:
                return action

        if destination_city:
            condition = slots.get("activity_condition")
            if weather and weather.get("status") == "OK":
                condition = weather.get("condition") or condition
            action = candidate(
                "recommend_activities",
                {
                    "city": destination_city,
                    "condition": condition or "",
                    "max_price": slots.get("activity_budget"),
                    "activity_type": slots.get("activity_type"),
                },
            )
            if action:
                return action

        if destination_city:
            action = candidate(
                "estimate_local_transport",
                {
                    "city": destination_city,
                    "transport_type": slots.get("transport_type") or "taxi",
                },
            )
            if action:
                return action

        if "calculate_total_cost" in required and "calculate_total_cost" not in called_tools:
            action = self._build_total_cost_action(slots, observations)
            if action and not self._is_duplicate_action(action, called_signatures):
                return action

        return None

    def _required_clarification(self, user_input: str) -> Optional[str]:
        slots = self._extract_slots(user_input)
        required = self._required_tool_names(user_input, slots)

        if "search_trips" in required:
            if not slots.get("date"):
                return "Bạn muốn đi ngày nào? Vui lòng cung cấp ngày theo định dạng YYYY-MM-DD."
            if not slots.get("origin"):
                return "Bạn muốn khởi hành từ thành phố hoặc sân bay nào?"
            if not slots.get("destination"):
                return "Bạn muốn đến thành phố hoặc sân bay nào?"

        if "get_weather" in required and not slots.get("date"):
            return "Bạn muốn kiểm tra thời tiết cho ngày nào? Vui lòng cung cấp ngày theo định dạng YYYY-MM-DD."

        return None

    def _has_required_observations(
        self,
        user_input: str,
        observations: List[Dict[str, Any]],
    ) -> bool:
        required = self._required_tool_names(user_input, self._extract_slots(user_input))
        called = {item["tool"] for item in observations}
        return required.issubset(called)

    def _required_tool_names(self, user_input: str, slots: Dict[str, Any]) -> set:
        text = self._normalize(user_input)
        raw_text = str(user_input or "").lower()
        required = set()

        ticket_signal = "vé" in raw_text or "ve may bay" in text
        has_route_slots = bool(slots.get("origin") and slots.get("destination"))
        route_signal = has_route_slots or ticket_signal or "->" in user_input or any(
            word in text for word in ["chuyen", "han", "sgn", "dad", "cxr", "pqc", "hui"]
        )
        if route_signal and (
            has_route_slots
            or "thoi tiet" not in text
            or "chuyen" in text
            or ticket_signal
            or "->" in user_input
        ):
            required.add("search_trips")

        if any(word in text for word in ["thoi tiet", "mua", "troi"]):
            required.add("get_weather")

        if any(word in text for word in ["mang gi", "mac gi", "trang phuc", "do mang theo"]):
            required.add("recommend_outfit")

        if "khach san" in text:
            required.add("search_hotels")

        if any(word in text for word in ["hoat dong", "di bien", "ngoai troi"]):
            required.add("recommend_activities")

        if any(word in text for word in ["taxi", "san bay", "trung tam", "di chuyen"]):
            required.add("estimate_local_transport")

        if any(word in text for word in ["tong chi phi", "tong tien", "tinh tien", "tinh tong", "chi phi co ban"]):
            required.add("calculate_total_cost")

        if "tinh" in text and "search_trips" in required and (
            "search_hotels" in required or "estimate_local_transport" in required
        ):
            required.add("calculate_total_cost")

        # If the user asks for rain-aware activities, activity recommendation should use weather first.
        if "recommend_activities" in required and any(word in text for word in ["mua", "nang", "thoi tiet"]):
            required.add("get_weather")

        return required

    def _extract_slots(self, user_input: str) -> Dict[str, Any]:
        text = self._normalize(user_input)
        slots: Dict[str, Any] = {}

        date_match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", user_input)
        if date_match:
            slots["date"] = date_match.group(0)

        route_match = re.search(r"\b([A-Z]{3})\s*->\s*([A-Z]{3})\b", user_input)
        if route_match:
            slots["origin"] = route_match.group(1)
            slots["destination"] = route_match.group(2)
            slots["origin_city"] = self._city_for_code(slots["origin"])
            slots["destination_city"] = self._city_for_code(slots["destination"])
        else:
            mentions = self._city_mentions(user_input)
            if len(mentions) >= 2 and any(word in text for word in ["tu ", "den", "vao", "ra"]):
                slots["origin"] = mentions[0]["code"]
                slots["destination"] = mentions[1]["code"]
                slots["origin_city"] = mentions[0]["city"]
                slots["destination_city"] = mentions[1]["city"]
            elif mentions:
                slots["city"] = mentions[0]["city"]
                slots["destination"] = mentions[0]["code"]
                slots["destination_city"] = mentions[0]["city"]

        passenger_match = re.search(r"(\d+)\s*(nguoi|khach|hanh khach)", text)
        if passenger_match:
            slots["passengers"] = int(passenger_match.group(1))

        room_match = re.search(r"(\d+)\s*(phong)", text)
        if room_match:
            slots["rooms"] = int(room_match.group(1))

        slots.update(self._extract_budgets(user_input))

        if "taxi" in text:
            slots["transport_type"] = "taxi"
        elif "bus" in text or "xe buyt" in text:
            slots["transport_type"] = "bus"
        elif "ride" in text or "grab" in text:
            slots["transport_type"] = "ride_hailing"

        if "ngoai troi" in text or "di bien" in text:
            slots["activity_type"] = "outdoor"
        elif "trong nha" in text:
            slots["activity_type"] = "indoor"

        if "mua" in text:
            slots["activity_condition"] = "rainy"
        elif any(phrase in text for phrase in ["troi nang", "co nang", "nang nong"]):
            slots["activity_condition"] = "sunny"

        return slots

    def _extract_budgets(self, user_input: str) -> Dict[str, Optional[int]]:
        text = self._normalize(user_input)
        budgets: Dict[str, Optional[int]] = {
            "flight_budget": None,
            "hotel_budget": None,
            "activity_budget": None,
        }
        pattern = re.compile(r"(\d+(?:[.,]\d+)?)\s*(trieu|nghin|k)")

        for match in pattern.finditer(text):
            amount = float(match.group(1).replace(",", "."))
            unit = match.group(2)
            value = int(amount * 1_000_000) if unit == "trieu" else int(amount * 1_000)
            before = text[max(0, match.start() - 55) : match.start()]
            after = text[match.end() : match.end() + 25]
            if "khach san" in before or "dem" in before or "dem" in after:
                budgets["hotel_budget"] = value
            elif "hoat dong" in before:
                budgets["activity_budget"] = value
            elif budgets["flight_budget"] is None:
                budgets["flight_budget"] = value

        return budgets

    def _build_total_cost_action(
        self,
        slots: Dict[str, Any],
        observations: List[Dict[str, Any]],
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        trip = self._first_trip(observations)
        transport = self._latest_observation(observations, "estimate_local_transport")
        hotel = self._first_hotel(observations)
        activity = self._first_activity(observations)

        if not trip and not transport and not hotel and not activity:
            return None

        args = {
            "ticket_price": trip.get("price", 0) if trip else 0,
            "passengers": slots.get("passengers") or 1,
            "baggage_fee": trip.get("baggage_fee", 0) if trip else 0,
            "hotel_price_per_night": hotel.get("price_per_night", 0) if hotel else 0,
            "nights": 1 if hotel else 0,
            "rooms": slots.get("rooms") or 1,
            "transport_cost": transport.get("cost", 0) if transport else 0,
            "activity_price": activity.get("price", 0) if activity else 0,
        }
        return "calculate_total_cost", args

    def _synthesize_final_answer(
        self,
        user_input: str,
        observations: List[Dict[str, Any]],
    ) -> str:
        if not observations:
            return "Mình chưa có đủ dữ liệu để trả lời. Bạn vui lòng bổ sung thông tin còn thiếu."

        parts = []
        for item in observations:
            tool = item["tool"]
            result = item["result"]
            status = result.get("status")

            if tool == "search_trips":
                parts.append(self._summarize_trips(result))
            elif tool == "get_weather":
                parts.append(self._summarize_weather(result))
            elif tool == "recommend_outfit":
                parts.append(self._summarize_outfit(result))
            elif tool == "search_hotels":
                parts.append(self._summarize_hotels(result))
            elif tool == "recommend_activities":
                parts.append(self._summarize_activities(result))
            elif tool == "estimate_local_transport":
                parts.append(self._summarize_transport(result))
            elif tool == "calculate_total_cost" and status == "OK":
                parts.append(self._summarize_total(result))

        answer = "\n\n".join(part for part in parts if part).strip()
        return answer or "Không có dữ liệu phù hợp trong bộ dữ liệu hiện có."

    def _summarize_trips(self, result: Dict[str, Any]) -> str:
        status = result.get("status")
        budget_text = ""
        if result.get("max_price"):
            budget_text = f" dưới {result.get('max_price'):,} VND/vé".replace(",", ".")

        if status == "OK":
            lines = [
                f"Các chuyến {result.get('origin')} -> {result.get('destination')} ngày {result.get('date')}{budget_text} phù hợp:"
            ]
            for trip in result.get("results", []):
                lines.append(
                    "- {provider} {flight_no}: {departure_time}-{arrival_time}, "
                    "{price:,} VND/vé, còn {seats_left} ghế, phí hành lý {baggage_fee:,} VND.".format(
                        **trip
                    ).replace(",", ".")
                )
            return "\n".join(lines)
        if status == "NO_MATCHING_TRIP":
            cheapest = result.get("all_route_options", [{}])[0]
            extra = ""
            if cheapest:
                extra = f" Chuyến rẻ nhất hiện có là {cheapest.get('provider')} {cheapest.get('flight_no')} giá {cheapest.get('price'):,} VND.".replace(",", ".")
            return f"Không có chuyến {result.get('origin')} -> {result.get('destination')} ngày {result.get('date')}{budget_text} phù hợp ngân sách.{extra}"
        if status == "NOT_ENOUGH_SEATS":
            option = (result.get("seat_limited_options") or [{}])[0]
            if option:
                return (
                    f"Không có chuyến {result.get('origin')} -> {result.get('destination')} ngày {result.get('date')}"
                    f"{budget_text} hợp lệ cho {result.get('passengers')} người. "
                    f"Chuyến gần nhất là {option.get('provider')} {option.get('flight_no')} giá "
                    f"{option.get('price'):,} VND nhưng chỉ còn {option.get('seats_left')} ghế, "
                    f"không đủ {result.get('passengers')} ghế."
                ).replace(",", ".")
            return f"Không có chuyến đúng ngân sách đủ {result.get('passengers')} ghế cho yêu cầu này."
        if status == "NO_ROUTE":
            return f"Không có route {result.get('origin')} -> {result.get('destination')} ngày {result.get('date')} trong dữ liệu."
        return result.get("message", "Không có dữ liệu chuyến bay phù hợp.")

    def _summarize_weather(self, result: Dict[str, Any]) -> str:
        if result.get("status") == "OK":
            return (
                f"Thời tiết {result.get('city')} ngày {result.get('date')}: "
                f"{result.get('condition')}, nhiệt độ {result.get('temp_low_c')}-{result.get('temp_high_c')}°C, "
                f"xác suất mưa {int(float(result.get('rain_probability', 0)) * 100)}%."
            )
        return f"Không có dữ liệu thời tiết cho {result.get('city')} ngày {result.get('date')}."

    def _summarize_outfit(self, result: Dict[str, Any]) -> str:
        if result.get("status") != "OK":
            return ""
        return f"Gợi ý đồ mang theo: {result.get('summary')}."

    def _summarize_hotels(self, result: Dict[str, Any]) -> str:
        budget_text = ""
        if result.get("max_price_per_night"):
            budget_text = f" dưới {result.get('max_price_per_night'):,} VND/đêm".replace(",", ".")

        if result.get("status") == "OK":
            lines = [f"Khách sạn phù hợp tại {result.get('city')}{budget_text}:"]
            for hotel in result.get("results", [])[:3]:
                lines.append(
                    f"- {hotel.get('hotel_name')}: {hotel.get('stars')} sao, {hotel.get('price_per_night'):,} VND/đêm, còn {hotel.get('available_rooms')} phòng.".replace(",", ".")
                )
            return "\n".join(lines)
        return f"Không có khách sạn tại {result.get('city')} phù hợp ngân sách/phòng yêu cầu."

    def _summarize_activities(self, result: Dict[str, Any]) -> str:
        budget_text = ""
        if result.get("max_price") is not None:
            budget_text = f" dưới {result.get('max_price'):,} VND".replace(",", ".")

        if result.get("status") == "OK":
            lines = [f"Hoạt động phù hợp tại {result.get('city')}{budget_text}:"]
            for activity in result.get("results", [])[:3]:
                lines.append(
                    f"- {activity.get('activity_name')} ({activity.get('type')}): {activity.get('price'):,} VND, khoảng {activity.get('duration_hours')} giờ.".replace(",", ".")
                )
            excluded = result.get("excluded_by_price") or []
            if excluded and result.get("max_price") is not None:
                lines.append("Loại vì vượt ngân sách:")
                for activity in excluded[:2]:
                    lines.append(
                        f"- {activity.get('activity_name')}: {activity.get('price'):,} VND.".replace(",", ".")
                    )
            return "\n".join(lines)
        return f"Không có hoạt động tại {result.get('city')} phù hợp điều kiện yêu cầu."

    def _summarize_transport(self, result: Dict[str, Any]) -> str:
        if result.get("status") == "OK":
            return (
                f"Di chuyển từ {result.get('from_airport')} về trung tâm bằng {result.get('transport_type')}: "
                f"{result.get('cost'):,} VND, khoảng {result.get('duration_minutes')} phút."
            ).replace(",", ".")
        return result.get("message", "Không có dữ liệu di chuyển nội đô.")

    def _summarize_total(self, result: Dict[str, Any]) -> str:
        items = result.get("line_items", {})
        return (
            "Tổng chi phí cơ bản: {total:,} VND "
            "(vé {ticket_total:,}, hành lý {baggage_total:,}, khách sạn {hotel_total:,}, "
            "di chuyển {transport_total:,}, hoạt động {activity_total:,})."
        ).format(total=result.get("total", 0), **items).replace(",", ".")

    def _latest_observation(
        self,
        observations: List[Dict[str, Any]],
        tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        for item in reversed(observations):
            if item["tool"] == tool_name:
                return item["result"]
        return None

    def _first_trip(self, observations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        result = self._latest_observation(observations, "search_trips")
        if result and result.get("status") == "OK" and result.get("results"):
            return result["results"][0]
        return None

    def _first_hotel(self, observations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        result = self._latest_observation(observations, "search_hotels")
        if result and result.get("status") == "OK" and result.get("results"):
            return result["results"][0]
        return None

    def _first_activity(self, observations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        result = self._latest_observation(observations, "recommend_activities")
        if result and result.get("status") == "OK" and result.get("results"):
            return result["results"][0]
        return None

    def _load_city_aliases(self) -> List[Dict[str, Any]]:
        with (DATA_DIR / "city_aliases.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda row: len(row["alias"]), reverse=True)
        return rows

    def _city_mentions(self, user_input: str) -> List[Dict[str, Any]]:
        text = self._normalize(user_input)
        mentions = []
        seen = set()
        for row in self._city_aliases:
            alias = self._normalize(row["alias"])
            start = self._find_alias_start(text, alias)
            if start < 0:
                continue
            key = (row["code"], start)
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                {
                    "code": row["code"],
                    "city": row["city"],
                    "airport_name": row["airport_name"],
                    "start": start,
                }
            )
        mentions.sort(key=lambda item: item["start"])
        deduped = []
        seen_codes = set()
        for item in mentions:
            if item["code"] in seen_codes:
                continue
            seen_codes.add(item["code"])
            deduped.append(item)
        return deduped

    def _find_alias_start(self, text: str, alias: str) -> int:
        if not alias:
            return -1

        # Short airport/city aliases such as "hn" and "dn" need word boundaries
        # so they are not accidentally matched inside longer words. Aliases with
        # punctuation or spaces, such as "tp.hcm", are safer with substring match.
        if alias.isalnum() and len(alias) <= 3:
            match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text)
            return match.start() if match else -1

        return text.find(alias)

    def _city_for_code(self, code: str) -> str:
        for row in self._city_aliases:
            if row["code"] == code:
                return row["city"]
        return code

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(float(str(value).replace(",", ".")))
        except (TypeError, ValueError):
            return None

    def _normalize(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.replace("đ", "d")
