import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgent:
    """
    A minimal text-ReAct agent for the lab.

    The agent asks the LLM for either:
    - an Action JSON object, then executes the selected tool, or
    - a Final Answer when enough evidence is available.
    """

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 6):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        self.last_run_trace: List[Dict[str, Any]] = []
        self.last_error_code: Optional[str] = None
        self.last_steps = 0

    def get_system_prompt(self) -> str:
        tool_descriptions = []
        for tool in self.tools:
            args = ", ".join(tool.get("args", []))
            tool_descriptions.append(
                f"- {tool['name']}({args}): {tool['description']}"
            )

        return f"""
You are Vietnam Trip Planner Agent, a ReAct travel assistant.

Your job:
- Break the user request into small steps.
- Use tools when you need trip, weather, hotel, activity, transport, or cost data.
- Use only observations returned by tools for factual trip/weather/hotel/price data.
- If required information is missing, ask a concise clarification question.
- If a tool returns no data, say that the data is unavailable instead of inventing it.
- Stop once you have enough evidence to answer.

Available tools:
{chr(10).join(tool_descriptions)}

Output format:
Use exactly one of these formats per turn.

Thought: short reason for the next step.
Action: {{"tool": "tool_name", "args": {{"arg_name": "value"}}}}

or

Thought: short reason why you can answer.
Final Answer: final response to the user.

Rules:
- Do not produce Observation yourself. The program will add Observation after tool execution.
- Do not wrap Action JSON in markdown fences.
- Do not call a tool that is not listed.
- Dates must use YYYY-MM-DD when calling tools.
""".strip()

    def run(self, user_input: str) -> str:
        logger.log_event(
            "AGENT_START",
            {"input": user_input, "model": self.llm.model_name, "max_steps": self.max_steps},
        )

        self.history = []
        self.last_run_trace = []
        self.last_error_code = None
        self.last_steps = 0

        scratchpad = ""

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
                },
            )

            final_answer = self._parse_final_answer(content)
            if final_answer:
                self.history.append({"role": "assistant", "content": content})
                self.last_run_trace.append(
                    {
                        "step": step,
                        "type": "final_answer",
                        "content": final_answer,
                    }
                )
                logger.log_event(
                    "AGENT_END",
                    {"status": "OK", "steps": step, "error_code": None},
                )
                return final_answer

            try:
                tool_name, tool_args = self._parse_action(content)
            except ValueError as exc:
                self.last_error_code = "PARSE_ERROR"
                observation = {
                    "status": "PARSE_ERROR",
                    "message": str(exc),
                    "expected_format": 'Action: {"tool": "tool_name", "args": {...}}',
                }
                logger.log_event(
                    "PARSE_ERROR",
                    {"step": step, "content": content, "message": str(exc)},
                )
                scratchpad += self._format_turn(content, observation)
                self.last_run_trace.append(
                    {
                        "step": step,
                        "type": "parse_error",
                        "content": content,
                        "observation": observation,
                    }
                )
                continue

            tool_result = self._execute_tool(tool_name, tool_args)
            status = tool_result.get("status") if isinstance(tool_result, dict) else None
            if status and status != "OK":
                self.last_error_code = status

            scratchpad += self._format_turn(content, tool_result)
            self.last_run_trace.append(
                {
                    "step": step,
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": tool_args,
                    "observation": tool_result,
                }
            )

        self.last_error_code = self.last_error_code or "MAX_STEPS_EXCEEDED"
        logger.log_event(
            "AGENT_END",
            {
                "status": "MAX_STEPS_EXCEEDED",
                "steps": self.max_steps,
                "error_code": self.last_error_code,
            },
        )
        return "Stopped: max steps reached before a final answer."

    def _build_prompt(self, user_input: str, scratchpad: str) -> str:
        if not scratchpad:
            scratchpad = "(no previous steps)"
        return f"""
User request:
{user_input}

Trace so far:
{scratchpad}

Choose the next Action or provide Final Answer.
""".strip()

    def _format_turn(self, llm_content: str, observation: Any) -> str:
        observation_text = json.dumps(observation, ensure_ascii=False)
        return f"{llm_content}\nObservation: {observation_text}\n\n"

    def _parse_final_answer(self, text: str) -> Optional[str]:
        match = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_action(self, text: str) -> Tuple[str, Dict[str, Any]]:
        action_text = self._extract_action_text(text)
        if not action_text:
            raise ValueError("No Action line found.")

        action_text = self._strip_code_fence(action_text)

        if action_text.startswith("{"):
            payload = json.loads(action_text)
            tool_name = payload.get("tool") or payload.get("name")
            args = payload.get("args") or payload.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if not tool_name:
                raise ValueError("Action JSON is missing 'tool'.")
            if not isinstance(args, dict):
                raise ValueError("Action args must be a JSON object.")
            return str(tool_name), args

        return self._parse_function_style_action(action_text)

    def _extract_action_text(self, text: str) -> str:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.strip().lower().startswith("action:"):
                action = line.split(":", 1)[1].strip()
                if action:
                    return action

                collected = []
                for next_line in lines[index + 1 :]:
                    if re.match(r"^\s*(Thought|Observation|Final Answer)\s*:", next_line, re.I):
                        break
                    collected.append(next_line)
                return "\n".join(collected).strip()
        return ""

    def _strip_code_fence(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _parse_function_style_action(self, text: str) -> Tuple[str, Dict[str, Any]]:
        try:
            parsed = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid function-style action: {text}") from exc

        call = parsed.body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError(f"Invalid function-style action: {text}")

        args: Dict[str, Any] = {}
        positional_args = [ast.literal_eval(arg) for arg in call.args]
        if positional_args:
            args["__positional_args__"] = positional_args

        for keyword in call.keywords:
            if keyword.arg is None:
                raise ValueError("Variadic keyword arguments are not supported.")
            args[keyword.arg] = ast.literal_eval(keyword.value)

        return call.func.id, args

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = next((item for item in self.tools if item["name"] == tool_name), None)
        if tool is None:
            result = {
                "status": "UNKNOWN_TOOL",
                "message": f"Tool {tool_name} not found.",
                "available_tools": [item["name"] for item in self.tools],
            }
            logger.log_event("UNKNOWN_TOOL", {"tool": tool_name, "args": args})
            return result

        func = tool.get("func")
        if func is None:
            result = {
                "status": "TOOL_NOT_CALLABLE",
                "message": f"Tool {tool_name} has no callable function.",
            }
            logger.log_event("TOOL_ERROR", {"tool": tool_name, "args": args, "result": result})
            return result

        logger.log_event("TOOL_CALL", {"tool": tool_name, "args": args})

        try:
            call_args = dict(args)
            positional_args = call_args.pop("__positional_args__", [])
            result = func(*positional_args, **call_args)
            if not isinstance(result, dict):
                result = {"status": "OK", "result": result}
        except TypeError as exc:
            result = {
                "status": "TOOL_ARGUMENT_ERROR",
                "message": str(exc),
                "tool": tool_name,
                "args": args,
            }
        except Exception as exc:  # noqa: BLE001 - lab telemetry should capture tool failures.
            result = {
                "status": "TOOL_RUNTIME_ERROR",
                "message": str(exc),
                "tool": tool_name,
            }

        logger.log_event("TOOL_RESULT", {"tool": tool_name, "result": result})
        return result
