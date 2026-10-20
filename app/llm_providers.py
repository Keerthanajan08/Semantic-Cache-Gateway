"""
Pluggable LLM providers behind one interface. Switching providers is a
config change (LLM_PROVIDER=mock|groq|gemini), not a code change: the
"AI gateway / multi-provider routing" pattern in miniature.

Every provider failure (timeout, rate limit, auth, malformed response) is
normalised to LLMProviderError, so the API layer can map it to one clean
502 without knowing which SDK was in use.
"""

import hashlib
import threading
import time
from typing import Any

from app.config import settings


class LLMProviderError(RuntimeError):
    """The upstream LLM failed or returned something unusable."""


class LLMConfigurationError(ValueError):
    """The selected provider can't be constructed (e.g. missing API key)."""


class BaseLLMProvider:
    name = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockProvider(BaseLLMProvider):
    """
    Deterministic, offline stand-in for a real LLM: same prompt, same answer,
    no API key, no network. Used for local development, demos and tests.
    """

    name = "mock"
    simulated_latency_s = 0.05  # so demo latencies resemble a (fast) real call

    def generate(self, prompt: str) -> str:
        time.sleep(self.simulated_latency_s)
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return (
            f"[mock-llm:{digest}] Generated answer based on the prompt/context "
            f"provided. Set LLM_PROVIDER=groq or gemini for a real response."
        )


def _require_key(value: str, env_var: str) -> str:
    if not value:
        raise LLMConfigurationError(f"{env_var} must be set to use this provider")
    return value


def _non_empty(text: str | None, provider: str) -> str:
    if not text or not text.strip():
        raise LLMProviderError(f"{provider} returned an empty response")
    return text


class GroqProvider(BaseLLMProvider):
    name = "groq"

    def __init__(self) -> None:
        from groq import Groq

        self.client: Any = Groq(
            api_key=_require_key(settings.groq_api_key, "GROQ_API_KEY"),
            timeout=settings.llm_timeout_seconds,
            max_retries=2,  # SDK retries 429/5xx with exponential backoff
        )
        self.model = settings.groq_model

    def generate(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.llm_max_output_tokens,
            )
            text = completion.choices[0].message.content
        except Exception as exc:
            raise LLMProviderError(f"groq request failed: {exc}") from exc
        return _non_empty(text, self.name)


class GeminiProvider(BaseLLMProvider):
    """Uses the current google-genai SDK (google-generativeai is deprecated)."""

    name = "gemini"

    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client: Any = genai.Client(
            api_key=_require_key(settings.gemini_api_key, "GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000)),
        )
        self.model = settings.gemini_model

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    max_output_tokens=settings.llm_max_output_tokens
                ),
            )
            text = response.text
        except Exception as exc:
            raise LLMProviderError(f"gemini request failed: {exc}") from exc
        return _non_empty(text, self.name)


PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "mock": MockProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}

_provider_instance: BaseLLMProvider | None = None
_provider_lock = threading.Lock()


def get_llm_provider() -> BaseLLMProvider:
    global _provider_instance
    with _provider_lock:
        if _provider_instance is None:
            try:
                provider_cls = PROVIDERS[settings.llm_provider]
            except KeyError:
                raise LLMConfigurationError(
                    f"Unknown LLM_PROVIDER {settings.llm_provider!r}; "
                    f"expected one of {sorted(PROVIDERS)}"
                ) from None
            _provider_instance = provider_cls()
        return _provider_instance
