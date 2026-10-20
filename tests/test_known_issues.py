"""
Known bugs, written as strict xfail tests. They describe the behaviour the
gateway *should* have. When a bug is fixed the test starts passing, strict
mode turns that into a failure, and you delete the xfail marker.
"""
import pytest

from app.config import settings
from app.llm_providers import get_llm_provider


@pytest.mark.xfail(
    strict=True,
    reason="QueryRequest only has min_length=1, so '   ' reaches the LLM and gets cached",
)
@pytest.mark.parametrize("query", ["   ", "\n\t"])
def test_whitespace_only_query_is_rejected(client, llm, query):
    response = client.post("/query", json={"query": query})
    assert response.status_code == 422
    assert llm.prompts == []


@pytest.mark.xfail(
    strict=True,
    reason="get_llm_provider silently falls back to MockProvider on a typo like 'grok'",
)
def test_unknown_provider_fails_loudly(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "grok")
    with pytest.raises(ValueError):
        get_llm_provider()
