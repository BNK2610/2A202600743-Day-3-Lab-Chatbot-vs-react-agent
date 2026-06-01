from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _read_prompt(filename: str) -> str:
    return (PROMPT_DIR / filename).read_text(encoding="utf-8").strip()


TRAVEL_PLANNER_SHARED_PROMPT = _read_prompt("shared_task.txt")

CHATBOT_BASELINE_SYSTEM_PROMPT = "\n\n".join(
    [
        TRAVEL_PLANNER_SHARED_PROMPT,
        _read_prompt("chatbot_baseline.txt"),
    ]
).strip()


def build_react_agent_system_prompt(tool_descriptions: str) -> str:
    react_prompt = _read_prompt("react_agent.txt").replace(
        "{{TOOL_DESCRIPTIONS}}",
        tool_descriptions,
    )
    return "\n\n".join([TRAVEL_PLANNER_SHARED_PROMPT, react_prompt]).strip()


def build_react_agent_v2_system_prompt(tool_descriptions: str) -> str:
    react_prompt = _read_prompt("react_agent_v2.txt").replace(
        "{{TOOL_DESCRIPTIONS}}",
        tool_descriptions,
    )
    return "\n\n".join([TRAVEL_PLANNER_SHARED_PROMPT, react_prompt]).strip()
