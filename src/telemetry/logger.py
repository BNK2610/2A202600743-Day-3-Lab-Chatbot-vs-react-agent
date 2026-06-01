import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict


class IndustryLogger:
    """
    Structured JSONL logger for lab telemetry.

    - File logs are JSON lines in UTF-8: logs/YYYY-MM-DD.log
    - Console logs are compact summaries, so the terminal stays readable.
    """

    def __init__(self, name: str = "AI-Lab-Agent", log_dir: str = "logs"):
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        self.console_enabled = os.getenv("LOG_TO_CONSOLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }

        self.file_logger = logging.getLogger(f"{name}.json")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.propagate = False
        self.file_logger.handlers.clear()

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.file_logger.addHandler(file_handler)

        self.console_logger = logging.getLogger(f"{name}.console")
        self.console_logger.setLevel(logging.INFO)
        self.console_logger.propagate = False
        self.console_logger.handlers.clear()

        if self.console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self.console_logger.addHandler(console_handler)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a structured event as one JSON object per line."""
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "data": data,
        }
        self.file_logger.info(json.dumps(payload, ensure_ascii=False, default=str))

        if self.console_enabled:
            self.console_logger.info(self._summarize_event(event_type, data))

    def info(self, msg: str):
        self.file_logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "INFO",
                    "data": {"message": msg},
                },
                ensure_ascii=False,
                default=str,
            )
        )
        if self.console_enabled:
            self.console_logger.info(f"[INFO] {msg}")

    def error(self, msg: str, exc_info=True):
        self.file_logger.error(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "ERROR",
                    "data": {"message": msg},
                },
                ensure_ascii=False,
                default=str,
            ),
            exc_info=exc_info,
        )
        if self.console_enabled:
            self.console_logger.error(f"[ERROR] {msg}", exc_info=exc_info)

    def _summarize_event(self, event_type: str, data: Dict[str, Any]) -> str:
        if event_type in {"CHATBOT_START", "AGENT_START"}:
            return f"[{event_type}] model={data.get('model', '')}"
        if event_type == "LLM_METRIC":
            return (
                f"[LLM_METRIC] provider={data.get('provider', '')} "
                f"tokens={data.get('total_tokens', 0)} latency_ms={data.get('latency_ms', 0)}"
            )
        if event_type == "AGENT_LLM_RESPONSE":
            return f"[AGENT_LLM_RESPONSE] step={data.get('step')} latency_ms={data.get('latency_ms', 0)}"
        if event_type == "TOOL_CALL":
            return f"[TOOL_CALL] {data.get('tool')} args={data.get('args')}"
        if event_type == "TOOL_RESULT":
            result = data.get("result", {})
            status = result.get("status") if isinstance(result, dict) else ""
            return f"[TOOL_RESULT] {data.get('tool')} status={status}"
        if event_type in {"CHATBOT_END", "AGENT_END", "PARSE_ERROR", "UNKNOWN_TOOL"}:
            return f"[{event_type}] {data}"
        return f"[{event_type}]"


logger = IndustryLogger()
