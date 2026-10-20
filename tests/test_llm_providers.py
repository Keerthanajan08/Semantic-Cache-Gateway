"""
Provider tests. The Groq and Gemini SDKs are replaced with in-process stubs,
so these run offline and never spend API quota — they verify that our code
calls each SDK correctly and unwraps its response, not the SDK itself.
"""
import sys
import types
from types import SimpleNamespace

import pytest

import app.llm_providers as llm_module
from app.config import settings
from app.llm_providers import (
    BaseLLMProvider,
    GeminiProvider,
    GroqProvider,
    MockProvider,
    get_llm_provider,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)


@pytest.fixture
def fake_groq(monkeypatch):
    calls = {}

    class Groq:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            calls["create"] = kwargs
            message = SimpleNamespace(content="groq says hi")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    module = types.ModuleType("groq")
    module.Groq = Groq
    monkeypatch.setitem(sys.modules, "groq", module)
    return calls


@pytest.fixture
def fake_gemini(monkeypatch):
    calls = {}

    class GenerativeModel:
        def __init__(self, name):
            calls["model"] = name

        def generate_content(self, prompt):
            calls["prompt"] = prompt
            return SimpleNamespace(text="gemini says hi")

    genai = types.ModuleType("google.generativeai")
    genai.configure = lambda api_key: calls.__setitem__("api_key", api_key)
    genai.GenerativeModel = GenerativeModel
    google = types.ModuleType("google")
    google.generativeai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai)
    return calls


class TestMockProvider:
    def test_is_deterministic(self):
        p = MockProvider()
        assert p.generate("same prompt") == p.generate("same prompt")

    def test_different_prompts_give_different_answers(self):
        p = MockProvider()
        assert p.generate("prompt one") != p.generate("prompt two")

    def test_answer_is_clearly_labelled_as_mock(self):
        assert MockProvider().generate("x").startswith("[mock-llm:")

    def test_name(self):
        assert MockProvider().name == "mock"


class TestGroqProvider:
    def test_sends_prompt_as_single_user_message(self, fake_groq, monkeypatch):
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "groq_model", "llama-test")

        answer = GroqProvider().generate("What is 2+2?")

        assert answer == "groq says hi"
        assert fake_groq["api_key"] == "gsk_test"
        assert fake_groq["create"]["model"] == "llama-test"
        assert fake_groq["create"]["messages"] == [{"role": "user", "content": "What is 2+2?"}]
        assert fake_groq["create"]["max_tokens"] == 500


class TestGeminiProvider:
    def test_configures_key_model_and_returns_text(self, fake_gemini, monkeypatch):
        monkeypatch.setattr(settings, "gemini_api_key", "AIza_test")
        monkeypatch.setattr(settings, "gemini_model", "gemini-test")

        answer = GeminiProvider().generate("hello")

        assert answer == "gemini says hi"
        assert fake_gemini == {"api_key": "AIza_test", "model": "gemini-test", "prompt": "hello"}


class TestGetLLMProvider:
    def test_defaults_to_mock(self):
        assert isinstance(get_llm_provider(), MockProvider)

    def test_selects_groq(self, fake_groq, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        assert isinstance(get_llm_provider(), GroqProvider)

    def test_selects_gemini(self, fake_gemini, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "gemini")
        assert isinstance(get_llm_provider(), GeminiProvider)

    def test_is_a_singleton(self):
        assert get_llm_provider() is get_llm_provider()


def test_base_provider_is_abstract():
    with pytest.raises(NotImplementedError):
        BaseLLMProvider().generate("x")
