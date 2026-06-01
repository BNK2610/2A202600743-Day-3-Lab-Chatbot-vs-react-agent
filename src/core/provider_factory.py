import os
from pathlib import Path
from typing import Optional

from src.core.llm_provider import LLMProvider


def load_env_file(path: str = ".env") -> None:
    """
    Load simple KEY=VALUE pairs from .env without requiring python-dotenv.
    """
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _clean_env(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if not value or value.startswith("your_"):
        return None
    return value


def create_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLMProvider:
    """
    Build the configured LLM provider from environment variables.
    """
    load_env_file()

    provider = (provider_name or os.getenv("DEFAULT_PROVIDER", "openai")).strip().lower()
    model = model_name or os.getenv("DEFAULT_MODEL")

    if provider == "openai":
        from src.core.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model_name=model or "gpt-4o",
            api_key=_clean_env(os.getenv("OPENAI_API_KEY")),
        )

    if provider == "openrouter":
        from src.core.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(
            model_name=model or "openai/gpt-4o-mini",
            api_key=_clean_env(os.getenv("OPENROUTER_API_KEY")),
        )

    if provider in {"endpoint", "llm_endpoint", "compatible"}:
        from src.core.llm_endpoint_provider import LLMEndpointProvider

        return LLMEndpointProvider(
            model_name=model,
            api_key=_clean_env(os.getenv("LLM_API_KEY")),
            endpoint=_clean_env(os.getenv("LLM_ENDPOINT")),
            provider_name=_clean_env(os.getenv("LLM_PROVIDER_NAME")) or "endpoint",
        )

    if provider in {"google", "gemini"}:
        from src.core.gemini_provider import GeminiProvider

        return GeminiProvider(
            model_name=model or "gemini-1.5-flash",
            api_key=_clean_env(os.getenv("GEMINI_API_KEY")),
        )

    if provider == "local":
        from src.core.local_provider import LocalProvider

        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        return LocalProvider(model_path=model_path)

    supported = "openai, openrouter, endpoint, google/gemini, local"
    raise ValueError(f"Unsupported DEFAULT_PROVIDER={provider!r}. Supported: {supported}")
