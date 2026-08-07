"""
Pluggable LLM providers behind one interface. Swapping providers is a
config change (LLM_PROVIDER=groq|gemini|mock), not a code change — this is
the "AI gateway / multi-provider routing" pattern in miniature.

MockProvider needs no API key and no network access at all, so the whole
project can be built, run, and demoed for $0 — real providers are a
drop-in swap once you have a free-tier key.
"""
import hashlib
import time

from app.config import settings


class BaseLLMProvider:
    name = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockProvider(BaseLLMProvider):
    """
    Deterministic, dependency-free stand-in for a real LLM. Useful for:
      - local development without an API key
      - unit tests (same input -> same output, no network flakiness)
      - proving out the cache/RAG logic in isolation before wiring a
        real model in
    """

    name = "mock"

    def generate(self, prompt: str) -> str:
        # Tiny artificial "thinking time" so latency numbers in demos look
        # like a real (if fast) LLM call rather than a suspicious 0ms.
        time.sleep(0.05)
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return (
            f"[mock-llm:{digest}] Generated answer based on the prompt/context "
            f"provided. Swap LLM_PROVIDER to 'groq' or 'gemini' in .env for a "
            f"real response."
        )


class GroqProvider(BaseLLMProvider):
    name = "groq"

    def __init__(self):
        from groq import Groq

        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return completion.choices[0].message.content


class GeminiProvider(BaseLLMProvider):
    name = "gemini"

    def __init__(self):
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


_provider_instance: BaseLLMProvider | None = None


def get_llm_provider() -> BaseLLMProvider:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    provider_map = {
        "mock": MockProvider,
        "groq": GroqProvider,
        "gemini": GeminiProvider,
    }
    provider_cls = provider_map.get(settings.llm_provider, MockProvider)
    _provider_instance = provider_cls()
    return _provider_instance
