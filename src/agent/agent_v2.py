import inspect
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agent.agent import ReActAgent
from src.core.llm_provider import LLMProvider
from src.core.prompts import build_react_agent_v2_system_prompt
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgentV2(ReActAgent):
    """
    LLM-first ReAct agent with lightweight validation guardrails.

    Compared with v1, this version does not decide the tool path with keyword
    rules. The LLM still owns Thought -> Action -> Observation decisions; code
    only validates action format, tool names, required args, duplicate calls,
    and premature final answers.
    """

    version = "v2"

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 8):
        super().__init__(llm=llm, tools=tools, max_steps=max_steps)
        self.tool_names = {tool["name"] for tool in tools}
        self.required_args_by_tool = {
            tool["name"]: self._required_args_for_tool(tool) for tool in tools
        }

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

            validation_error = self._validate_no_hallucinated_observation(content)
            if validation_error:
                scratchpad += self._format_turn(content, validation_error)
                self._record_validation_error(step, content, validation_error)
                continue

            final_answer = self._parse_final_answer(content)
            if final_answer:
                is_need_info = self._is_need_info_answer(final_answer)
                validation_error = self._validate_final_answer(
                    final_answer=final_answer,
                    observations=observations,
                    is_need_info=is_need_info,
                )
                if validation_error:
                    scratchpad += self._format_turn(content, validation_error)
                    self._record_validation_error(step, content, validation_error)
                    continue

                if is_need_info:
                    final_answer = self._strip_need_info_marker(final_answer)

                if not observations and is_need_info:
                    self.last_error_code = "MISSING_REQUIRED_INFO"

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
            if action is None:
                observation = {
                    "status": "PARSE_ERROR",
                    "message": "Output must contain either Action JSON or Final Answer.",
                    "expected_action_format": 'Action: {"tool": "tool_name", "args": {...}}',
                    "expected_final_format": "Final Answer: ...",
                }
                scratchpad += self._format_turn(content, observation)
                self._record_validation_error(step, content, observation)
                continue

            tool_name, tool_args = action
            validation_error = self._validate_action(
                tool_name=tool_name,
                tool_args=tool_args,
                called_signatures=called_signatures,
            )
            if validation_error:
                scratchpad += self._format_turn(content, validation_error)
                self._record_validation_error(step, content, validation_error)
                continue

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
        answer = self._force_final_answer(user_input, scratchpad, observations)
        logger.log_event(
            "AGENT_END",
            {
                "status": "MAX_STEPS_WITH_FINALIZATION",
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

    def _parse_final_answer(self, text: str) -> Optional[str]:
        matches = list(
            re.finditer(
                r"(?im)^\s*Final Answer\s*:\s*(.+)",
                text or "",
                flags=re.DOTALL,
            )
        )
        if not matches:
            return None
        return matches[-1].group(1).strip()

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

    def _required_args_for_tool(self, tool: Dict[str, Any]) -> List[str]:
        func = tool.get("func")
        if not callable(func):
            return []

        signature = inspect.signature(func)
        required = []
        for name, parameter in signature.parameters.items():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
        return required

    def _validate_action(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        called_signatures: set,
    ) -> Optional[Dict[str, Any]]:
        if tool_name not in self.tool_names:
            return {
                "status": "VALIDATION_ERROR",
                "error_code": "UNKNOWN_TOOL",
                "message": f"Tool '{tool_name}' is not available. Choose one tool from available_tools.",
                "available_tools": sorted(self.tool_names),
            }

        missing_args = [
            arg
            for arg in self.required_args_by_tool.get(tool_name, [])
            if self._is_missing(tool_args.get(arg))
        ]
        if missing_args:
            return {
                "status": "VALIDATION_ERROR",
                "error_code": "MISSING_REQUIRED_ARGS",
                "tool": tool_name,
                "missing_args": missing_args,
                "provided_args": tool_args,
                "message": (
                    "The action is missing required arguments. If the user did not "
                    "provide these values, ask with Final Answer: NEED_INFO: <question>. "
                    "Otherwise, retry the Action with complete args."
                ),
            }

        signature = self._action_signature(tool_name, tool_args)
        if signature in called_signatures:
            return {
                "status": "VALIDATION_ERROR",
                "error_code": "DUPLICATE_ACTION",
                "tool": tool_name,
                "args": tool_args,
                "message": (
                    "This exact tool call was already executed. Use the previous "
                    "Observation, choose a different Action, or provide Final Answer."
                ),
            }

        return None

    def _validate_no_hallucinated_observation(self, content: str) -> Optional[Dict[str, Any]]:
        if not re.search(r"(?im)^\s*Observation\s*:", content or ""):
            return None

        return {
            "status": "VALIDATION_ERROR",
            "error_code": "HALLUCINATED_OBSERVATION",
            "message": (
                "The model output contains an Observation. Observations must only be "
                "created by the program after a real tool call. Choose an Action with "
                "the needed tool instead, or provide Final Answer using existing "
                "Observations only."
            ),
        }

    def _validate_final_answer(
        self,
        final_answer: str,
        observations: List[Dict[str, Any]],
        is_need_info: bool,
    ) -> Optional[Dict[str, Any]]:
        if not observations and is_need_info:
            return None

        if observations:
            return None

        return {
            "status": "VALIDATION_ERROR",
            "error_code": "FINAL_ANSWER_TOO_EARLY",
            "message": (
                "Final Answer before any tool Observation is only allowed for a "
                "clarification question. If the request needs CSV/tool data, choose "
                "an Action first. If required information is missing, ask with "
                "Final Answer: NEED_INFO: <question>."
            ),
        }

    def _record_validation_error(
        self,
        step: int,
        content: str,
        observation: Dict[str, Any],
    ) -> None:
        logger.log_event(
            "AGENT_V2_VALIDATION_ERROR",
            {
                "step": step,
                "content": content,
                "observation": observation,
                "agent_version": self.version,
            },
        )
        self.last_run_trace.append(
            {
                "step": step,
                "type": "validation_error",
                "content": content,
                "observation": observation,
            }
        )

    def _force_final_answer(
        self,
        user_input: str,
        scratchpad: str,
        observations: List[Dict[str, Any]],
    ) -> str:
        if not observations:
            return "Mình chưa có đủ dữ liệu để trả lời. Bạn vui lòng bổ sung thông tin còn thiếu."

        prompt = f"""
User request:
{user_input}

Trace so far:
{scratchpad}

The agent reached max steps. Provide a concise Final Answer using only the Observations above.
If the Observations show missing data, say that clearly. Do not call another tool.
""".strip()
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
                "step": "finalize",
                "content": content,
                "latency_ms": result.get("latency_ms", 0),
                "agent_version": self.version,
            },
        )

        final_answer = self._parse_final_answer(content) or content
        self.last_run_trace.append(
            {
                "step": self.max_steps,
                "type": "final_answer",
                "source": "max_steps_finalization",
                "content": final_answer,
            }
        )
        return final_answer

    def _action_signature(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        return json.dumps(
            {"tool": tool_name, "args": tool_args},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _is_missing(self, value: Any) -> bool:
        return value is None or value == "" or value == []

    def _is_need_info_answer(self, answer: str) -> bool:
        return bool(re.match(r"(?is)^\s*(?:\[?NEED_INFO\]?\s*:)", answer or ""))

    def _strip_need_info_marker(self, answer: str) -> str:
        return re.sub(r"(?is)^\s*(?:\[?NEED_INFO\]?\s*:)\s*", "", answer or "").strip()
