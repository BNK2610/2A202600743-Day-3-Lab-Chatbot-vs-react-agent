import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "report" / "evaluation_results"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def compact_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def split_ids(ids: Optional[str]) -> Optional[set]:
    if not ids:
        return None
    return {item.strip() for item in ids.split(",") if item.strip()}


def extract_flight_codes(text: str) -> List[str]:
    return re.findall(r"\b(?:VJ|VN|QH)\d+\b", text or "")


def extract_big_numbers(text: str) -> List[str]:
    return [item for item in re.findall(r"\d+", text or "") if len(item) >= 4]


def answer_has_no_data_signal(answer: str) -> bool:
    normalized = normalize_text(answer)
    signals = [
        "khong co",
        "khong tim thay",
        "khong du lieu",
        "khong co du lieu",
        "chua co du lieu",
        "khong kha dung",
        "unavailable",
        "not available",
        "no matching",
        "no data",
        "not found",
    ]
    return any(signal in normalized for signal in signals)


def answer_has_clarification(answer: str) -> bool:
    normalized = normalize_text(answer)
    return "?" in str(answer) or any(
        phrase in normalized
        for phrase in [
            "ban muon",
            "vui long cho biet",
            "can cung cap",
            "ngay nao",
            "noi di",
            "origin",
            "date",
        ]
    )


def score_answer(
    answer: str,
    truth: Dict[str, str],
    called_tools: Optional[Iterable[str]] = None,
    check_tools: bool = False,
) -> Dict[str, Any]:
    """
    Lightweight heuristic scoring.

    This is not a replacement for human review; it produces a first-pass pass/fail
    signal for the report table.
    """
    answer_norm = normalize_text(answer)
    answer_digits = compact_digits(answer)
    expected_facts = truth.get("expected_facts", "")
    expected_error = truth.get("expected_error_code", "")
    required_tools = [
        item
        for item in (truth.get("required_tools") or "").split("|")
        if item and item != "none"
    ]
    called_tool_set = set(called_tools or [])

    reasons: List[str] = []
    passed = True

    if check_tools:
        missing_tools = [tool for tool in required_tools if tool not in called_tool_set]
        if missing_tools:
            passed = False
            reasons.append(f"missing tools: {', '.join(missing_tools)}")

    if expected_error in {"MISSING_REQUIRED_INFO"}:
        if not answer_has_clarification(answer):
            passed = False
            reasons.append("expected clarification question")
    elif expected_error:
        if not answer_has_no_data_signal(answer):
            passed = False
            reasons.append(f"expected no-data/error behavior: {expected_error}")

    for code in extract_flight_codes(expected_facts):
        if code not in answer:
            reasons.append(f"missing expected flight code {code}")

    # Only require the first two large numeric facts. This keeps the heuristic useful
    # without making it brittle to alternate correct wording.
    for number in extract_big_numbers(expected_facts)[:2]:
        if number not in answer_digits:
            reasons.append(f"missing expected number {number}")

    if not answer.strip():
        passed = False
        reasons.append("empty answer")

    if reasons and not expected_error:
        # Missing some facts usually needs human review, not an automatic fail.
        verdict = "review"
    else:
        verdict = "pass" if passed else "fail"

    return {
        "verdict": verdict,
        "reasons": "; ".join(reasons) if reasons else "basic checks passed",
    }


def called_tools_from_trace(trace: List[Dict[str, Any]]) -> List[str]:
    return [
        item.get("tool", "")
        for item in trace
        if item.get("type") == "tool_call" and item.get("tool")
    ]


def run_evaluation(args: argparse.Namespace) -> Path:
    scenarios = load_csv(DATA_DIR / "test_scenarios.csv")
    ground_truth = {
        row["id"]: row for row in load_csv(DATA_DIR / "eval_ground_truth.csv")
    }
    selected_ids = split_ids(args.ids)
    if selected_ids:
        scenarios = [row for row in scenarios if row["id"] in selected_ids]

    if args.limit:
        scenarios = scenarios[: args.limit]

    if args.dry_run:
        for row in scenarios:
            print(f"{row['id']} [{row['type']}/{row['difficulty']}]: {row['user_query']}")
        return Path("")

    from chatbot import ChatbotBaseline
    from src.core.provider_factory import create_llm_provider
    from src.tools.travel_tools import get_travel_tools

    if args.agent_version == "v2":
        from src.agent.agent_v2 import ReActAgentV2 as AgentClass
    else:
        from src.agent.agent_v1 import ReActAgentV1 as AgentClass

    llm = create_llm_provider(provider_name=args.provider, model_name=args.model)
    chatbot = ChatbotBaseline(llm)
    agent = AgentClass(llm=llm, tools=get_travel_tools(), max_steps=args.max_steps)

    rows: List[Dict[str, Any]] = []

    for scenario in scenarios:
        truth = ground_truth.get(scenario["id"], {})
        query = scenario["user_query"]
        print(f"Running {scenario['id']}: {query}")

        chatbot_answer = ""
        chatbot_latency_ms = 0
        chatbot_score = {"verdict": "skipped", "reasons": ""}
        if not args.agent_only:
            start = time.time()
            chatbot_answer = chatbot.run(query)
            chatbot_latency_ms = int((time.time() - start) * 1000)
            chatbot_score = score_answer(
                chatbot_answer,
                truth,
                called_tools=None,
                check_tools=False,
            )

        agent_answer = ""
        agent_latency_ms = 0
        agent_tools: List[str] = []
        agent_score = {"verdict": "skipped", "reasons": ""}
        if not args.chatbot_only:
            start = time.time()
            agent_answer = agent.run(query)
            agent_latency_ms = int((time.time() - start) * 1000)
            agent_tools = called_tools_from_trace(agent.last_run_trace)
            agent_score = score_answer(
                agent_answer,
                truth,
                called_tools=agent_tools,
                check_tools=True,
            )

        rows.append(
            {
                "id": scenario["id"],
                "type": scenario["type"],
                "difficulty": scenario["difficulty"],
                "user_query": query,
                "required_tools": truth.get("required_tools", ""),
                "expected_error_code": truth.get("expected_error_code", ""),
                "agent_version": args.agent_version,
                "chatbot_verdict": chatbot_score["verdict"],
                "chatbot_score_notes": chatbot_score["reasons"],
                "chatbot_latency_ms": chatbot_latency_ms,
                "chatbot_answer": chatbot_answer,
                "agent_verdict": agent_score["verdict"],
                "agent_score_notes": agent_score["reasons"],
                "agent_latency_ms": agent_latency_ms,
                "agent_loop_count": agent.last_steps,
                "agent_error_code": agent.last_error_code or "",
                "agent_tools_called": "|".join(agent_tools),
                "agent_answer": agent_answer,
                "agent_trace_json": json.dumps(agent.last_run_trace, ensure_ascii=False),
            }
        )

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_results_{timestamp}.csv"
    json_output_path = output_dir / f"eval_results_{timestamp}.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_rows = []
    for row in rows:
        json_row = dict(row)
        trace_json = json_row.pop("agent_trace_json", "[]")
        try:
            json_row["agent_trace"] = json.loads(trace_json)
        except json.JSONDecodeError:
            json_row["agent_trace"] = []
        json_rows.append(json_row)

    with json_output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scenario_count": len(json_rows),
                "results": json_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved evaluation results to: {output_path}")
    print(f"Saved readable JSON results to: {json_output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chatbot baseline vs ReAct Agent on CSV scenarios."
    )
    parser.add_argument("--provider", help="Override DEFAULT_PROVIDER from .env.")
    parser.add_argument("--model", help="Override DEFAULT_MODEL from .env.")
    parser.add_argument("--ids", help="Comma-separated scenario IDs, e.g. T01,T03,T21.")
    parser.add_argument("--limit", type=int, help="Limit number of scenarios.")
    parser.add_argument("--max-steps", type=int, default=8, help="Agent max ReAct steps.")
    parser.add_argument(
        "--agent-version",
        choices=["v1", "v2"],
        default="v1",
        help="Choose which agent implementation to evaluate.",
    )
    parser.add_argument("--agent-only", action="store_true", help="Skip chatbot baseline.")
    parser.add_argument("--chatbot-only", action="store_true", help="Skip ReAct agent.")
    parser.add_argument("--dry-run", action="store_true", help="Print scenarios without calling LLM.")
    parser.add_argument("--output-dir", help="Directory for evaluation CSV output.")
    args = parser.parse_args()

    if args.agent_only and args.chatbot_only:
        raise ValueError("Use at most one of --agent-only or --chatbot-only.")

    run_evaluation(args)


if __name__ == "__main__":
    main()
