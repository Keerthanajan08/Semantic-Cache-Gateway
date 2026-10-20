"""
Provider tests. The Groq and Gemini SDKs are replaced with in-process stubs,
so these run offline and never spend API quota: they verify that our code
calls each SDK correctly, unwraps its response and normalises its errors.
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
    LLMConfigurationError,
    LLMProviderError,
    MockProvider,
    get_llm_provider,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)


@pytest.fixture
def fake_groq(monkeypatch):
    state = {"reply": "groq says hi", "error": None}

    class Groq:
        def __init__(self, **kwargs):
            state["init"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            state["create"] = kwargs
            if state["error"]:
                raise state["error"]
            message = SimpleNamespace(content=state["reply"])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    module = types.ModuleType("groq")
    module.Groq = Groq
    monkeypatch.setitem(sys.modules, "groq", module)
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    return state


@pytest.fixture
def fake_gemini(monkeypatch):
    state = {"reply": "gemini says hi", "error": None}

    class Models:
        def generate_content(self, **kwargs):
            state["generate"] = kwargs
            if state["error"]:
                raise state["error"]
            return SimpleNamespace(text=state["reply"])

    class Client:
        def __init__(self, **kwargs):
            state["init"] = kwargs
            self.models = Models()

    gtypes = types.ModuleType("google.genai.types")
    gtypes.HttpOptions = lambda **kw: ("HttpOptions", kw)
    gtypes.GenerateContentConfig = lambda **kw: ("GenerateContentConfig", kw)
    genai = types.ModuleType("google.genai")
    genai.Client = Client
    genai.types = gtypes
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    monkeypatch.setattr(settings, "gemini_api_key", "AIza_test")
    return state


class TestMockProvider:
    def test_is_deterministic_and_prompt_sensitive(self):
        p = MockProvider()
        assert p.generate("same") == p.generate("same")
        assert p.generate("one") != p.generate("two")

    def test_answer_is_clearly_labelled_as_mock(self):
        assert MockProvider().generate("x").startswith("[mock-llm:")


class TestGroqProvider:
    def test_request_shape(self, fake_groq, monkeypatch):
        monkeypatch.setattr(settings, "groq_model", "openai/gpt-oss-20b")
        monkeypatch.setattr(settings, "llm_timeout_seconds", 12.5)
        monkeypatch.setattr(settings, "llm_max_output_tokens", 321)

        assert GroqProvider().generate("What is 2+2?") == "groq says hi"

        assert fake_groq["init"]["api_key"] == "gsk_test"
        assert fake_groq["init"]["timeout"] == 12.5
        assert fake_groq["create"]["model"] == "openai/gpt-oss-20b"
        assert fake_groq["create"]["messages"] == [{"role": "user", "content": "What is 2+2?"}]
        assert fake_groq["create"]["max_tokens"] == 321

    def test_sdk_errors_become_provider_errors(self, fake_groq):
        fake_groq["error"] = TimeoutError("read timed out")
        with pytest.raises(LLMProviderError, match="groq request failed: read timed out"):
            GroqProvider().generate("x")

    @pytest.mark.parametrize("reply", [None, "", "   "])
    def test_empty_reply_is_an_error(self, fake_groq, reply):
        fake_groq["reply"] = reply
        with pytest.raises(LLMProviderError, match="empty response"):
            GroqProvider().generate("x")

    def test_missing_key_fails_fast(self, fake_groq, monkeypatch):
        monkeypatch.setattr(settings, "groq_api_key", "")
        with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
            GroqProvider()


class TestGeminiProvider:
    def test_request_shape(self, fake_gemini, monkeypatch):
        monkeypatch.setattr(settings, "gemini_model", "gemini-3.5-flash-lite")
        monkeypatch.setattr(settings, "llm_timeout_seconds", 7)
        monkeypatch.setattr(settings, "llm_max_output_tokens", 99)

        assert GeminiProvider().generate("hello") == "gemini says hi"

        assert fake_gemini["init"]["api_key"] == "AIza_test"
        assert fake_gemini["init"]["http_options"] == ("HttpOptions", {"timeout": 7000})
        assert fake_gemini["generate"]["model"] == "gemini-3.5-flash-lite"
        assert fake_gemini["generate"]["contents"] == "hello"
        assert fake_gemini["generate"]["config"] == (
            "GenerateContentConfig",
            {"max_output_tokens": 99},
        )

    def test_sdk_errors_become_provider_errors(self, fake_gemini):
        fake_gemini["error"] = RuntimeError("429 RESOURCE_EXHAUSTED")
        with pytest.raises(LLMProviderError, match="RESOURCE_EXHAUSTED"):
            GeminiProvider().generate("x")

    def test_blocked_response_without_text_is_an_error(self, fake_gemini):
        fake_gemini["reply"] = None  # e.g. safety-blocked: .text is None
        with pytest.raises(LLMProviderError, match="empty response"):
            GeminiProvider().generate("x")

    def test_missing_key_fails_fast(self, fake_gemini, monkeypatch):
        monkeypatch.setattr(settings, "gemini_api_key", "")
        with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
            GeminiProvider()


class TestGetLLMProvider:
    def test_defaults_to_mock(self):
        assert isinstance(get_llm_provider(), MockProvider)

    @pytest.mark.parametrize(
        ("name", "fixture", "cls"),
        [("groq", "fake_groq", GroqProvider), ("gemini", "fake_gemini", GeminiProvider)],
    )
    def test_selects_configured_provider(self, request, monkeypatch, name, fixture, cls):
        request.getfixturevalue(fixture)
        monkeypatch.setattr(settings, "llm_provider", name)
        assert isinstance(get_llm_provider(), cls)

    def test_unknown_provider_fails_loudly(self, monkeypatch):
        """Regression: unknown names used to fall back to the mock silently."""
        monkeypatch.setattr(settings, "llm_provider", "grok")
        with pytest.raises(LLMConfigurationError, match="grok"):
            get_llm_provider()

    def test_is_a_singleton(self):
        assert get_llm_provider() is get_llm_provider()


def test_base_provider_is_abstract():
    with pytest.raises(NotImplementedError):
        BaseLLMProvider().generate("x")
