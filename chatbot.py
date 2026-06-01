import argparse
from typing import Any, Dict

from src.core.llm_provider import LLMProvider
from src.core.prompts import CHATBOT_BASELINE_SYSTEM_PROMPT
from src.core.provider_factory import create_llm_provider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ChatbotBaseline:
    """
    Plain LLM chatbot baseline. It has no access to tools.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.last_result: Dict[str, Any] = {}

    def run(self, user_input: str) -> str:
        logger.log_event(
            "CHATBOT_START",
            {"input": user_input, "model": self.llm.model_name},
        )

        result = self.llm.generate(user_input, system_prompt=CHATBOT_BASELINE_SYSTEM_PROMPT)
        self.last_result = result

        tracker.track_request(
            provider=result.get("provider", "unknown"),
            model=self.llm.model_name,
            usage=result.get("usage", {}),
            latency_ms=result.get("latency_ms", 0),
        )

        answer = (result.get("content") or "").strip()
        logger.log_event(
            "CHATBOT_END",
            {
                "latency_ms": result.get("latency_ms", 0),
                "usage": result.get("usage", {}),
                "answer": answer,
            },
        )
        return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the plain chatbot baseline.")
    parser.add_argument("question", nargs="*", help="Question to ask the chatbot.")
    parser.add_argument("--provider", help="Override DEFAULT_PROVIDER from .env.")
    parser.add_argument("--model", help="Override DEFAULT_MODEL from .env.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("User: ").strip()

    llm = create_llm_provider(provider_name=args.provider, model_name=args.model)
    chatbot = ChatbotBaseline(llm)
    print(chatbot.run(question))


if __name__ == "__main__":
    main()
