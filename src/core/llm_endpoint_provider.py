import os
import time
from typing import Dict, Any, Optional, Generator
from openai import OpenAI
from src.core.llm_provider import LLMProvider


class LLMEndpointProvider(LLMProvider):
    """
    Generic provider for OpenAI-compatible LLM endpoints.

    Useful for services such as OpenRouter, Groq, Together, DeepInfra,
    or local OpenAI-compatible servers.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        provider_name: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("LLM_API_KEY")
        endpoint = endpoint or os.getenv("LLM_ENDPOINT")
        provider_name = provider_name or os.getenv("LLM_PROVIDER_NAME", "llm_endpoint")
        model_name = model_name or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

        if not api_key:
            raise ValueError("LLM_API_KEY or api_key is required for LLMEndpointProvider")
        if not endpoint:
            raise ValueError("LLM_ENDPOINT or endpoint is required for LLMEndpointProvider")

        super().__init__(model_name, api_key)
        self.endpoint = endpoint
        self.provider_name = provider_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.endpoint)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response.choices[0].message.content
        usage_data = response.usage
        usage = {
            "prompt_tokens": usage_data.prompt_tokens if usage_data else 0,
            "completion_tokens": usage_data.completion_tokens if usage_data else 0,
            "total_tokens": usage_data.total_tokens if usage_data else 0,
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": self.provider_name,
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
