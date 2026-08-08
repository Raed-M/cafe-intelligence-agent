"""Provider-aware prompt caching.

The mechanism differs per provider and is unavailable on this project's current
Gemini free-tier key (explicit cache creation is refused with
`TotalCachedContentStorageTokensPerModelFreeTier ... limit=0`). These tests pin
the behaviour that matters either way: correct shape per provider, and a
degradation path that never drops the prompt or raises.
"""
import pytest

from src.tools import prompt_cache

PROMPT = "You are an assistant. " * 60


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    prompt_cache.reset_for_tests()
    monkeypatch.delenv("LLM_PROMPT_CACHE", raising=False)
    # Gemini explicit caching is opt-in (it costs a network round-trip that
    # cannot succeed on the free tier); the tests below that exercise that path
    # enable it deliberately.
    monkeypatch.setenv("LLM_PROMPT_CACHE_GEMINI_EXPLICIT", "1")
    yield
    prompt_cache.reset_for_tests()


def test_anthropic_prompt_is_marked_for_caching():
    """Anthropic caches nothing without an explicit marker, so this is the one
    provider where omitting the code means omitting the feature.

    It must be a SystemMessage, not a bare block list: create_react_agent takes
    `str | SystemMessage | callable`, and a raw list quietly changed how the
    prompt was applied rather than raising."""
    from langchain_core.messages import SystemMessage

    result = prompt_cache.cacheable_system_prompt(PROMPT, "anthropic")
    assert isinstance(result, SystemMessage)
    assert result.content == [
        {"type": "text", "text": PROMPT, "cache_control": {"type": "ephemeral"}}
    ]


@pytest.mark.parametrize("provider", ["openai", "gemini", "unknown"])
def test_other_providers_get_the_plain_prompt(provider):
    """OpenAI caches automatically and Gemini caches via cached_content; neither
    takes an inline marker, and an unmarked string must round-trip unchanged."""
    assert prompt_cache.cacheable_system_prompt(PROMPT, provider) == PROMPT


def test_gemini_explicit_caching_is_off_by_default(monkeypatch):
    """It must not fire a network request unless asked: on the free tier the
    request can only fail, and it would drag offline tests onto the network."""
    monkeypatch.delenv("LLM_PROMPT_CACHE_GEMINI_EXPLICIT", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def _explode(**kwargs):
        raise AssertionError("no provider call may be made when the flag is off")

    monkeypatch.setattr("google.genai.Client", _explode)
    assert prompt_cache.gemini_context_cache("gemini-3.6-flash", PROMPT) is None


def test_disable_switch_turns_off_every_path(monkeypatch):
    monkeypatch.setenv("LLM_PROMPT_CACHE", "0")
    assert prompt_cache.cacheable_system_prompt(PROMPT, "anthropic") == PROMPT
    assert prompt_cache.gemini_context_cache("gemini-3.6-flash", PROMPT) is None


def test_gemini_cache_refusal_degrades_to_none(monkeypatch):
    """A provider that refuses to create a cache (free tier, or a prompt below
    the minimum cacheable size) must yield None, not raise: the caller then
    sends the prompt normally."""
    class _Boom:
        def __init__(self, **kwargs):
            raise RuntimeError("429 TotalCachedContentStorageTokensPerModelFreeTier limit=0")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", _Boom)
    assert prompt_cache.gemini_context_cache("gemini-3.6-flash", PROMPT) is None


def test_refusal_is_remembered_so_it_is_attempted_once(monkeypatch):
    """Without memoization a free-tier key would fire a doomed cache-create
    request before every single chat turn."""
    attempts = []

    class _Boom:
        def __init__(self, **kwargs):
            attempts.append(1)
            raise RuntimeError("refused")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", _Boom)
    for _ in range(4):
        assert prompt_cache.gemini_context_cache("gemini-3.6-flash", PROMPT) is None
    assert len(attempts) == 1


def test_successful_cache_is_reused_not_recreated(monkeypatch):
    created = []

    class _Caches:
        def create(self, **kwargs):
            created.append(kwargs)
            return type("C", (), {"name": "cachedContents/abc"})()

    class _Client:
        def __init__(self, **kwargs):
            self.caches = _Caches()

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", _Client)
    first = prompt_cache.gemini_context_cache("gemini-3.6-flash", PROMPT)
    second = prompt_cache.gemini_context_cache("gemini-3.6-flash", PROMPT)
    assert first == second == "cachedContents/abc"
    assert len(created) == 1, "the cache must be created once and reused"
    assert created[0]["model"] == "gemini-3.6-flash"


def test_a_changed_prompt_never_reuses_the_old_cache(monkeypatch):
    """Cache keys include a prompt digest, so editing the system prompt cannot
    silently keep serving the previous one."""
    created = []

    class _Caches:
        def create(self, **kwargs):
            created.append(kwargs)
            return type("C", (), {"name": f"cachedContents/{len(created)}"})()

    class _Client:
        def __init__(self, **kwargs):
            self.caches = _Caches()

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", _Client)
    a = prompt_cache.gemini_context_cache("gemini-3.6-flash", PROMPT)
    b = prompt_cache.gemini_context_cache("gemini-3.6-flash", PROMPT + " edited")
    assert a != b
    assert len(created) == 2


def test_missing_api_key_yields_no_cache(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert prompt_cache.gemini_context_cache("gemini-3.6-flash", PROMPT) is None


def test_chat_still_sends_the_prompt_when_there_is_no_cache(monkeypatch):
    """The one regression that would be expensive and invisible: dropping the
    system prompt because a cache was assumed to exist."""
    from api.services import chat as chat_module

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr("src.tools.prompt_cache.gemini_context_cache", lambda *a, **k: None)
    monkeypatch.setattr(chat_module, "get_chat_model", lambda *a, **k: object())
    _, prompt = chat_module._cached_model_and_prompt()
    assert prompt == chat_module.SYSTEM_PROMPT


def test_chat_omits_the_prompt_only_when_it_is_cached(monkeypatch):
    """Conversely, when the prompt IS in the cache it must not be sent again --
    that would bill the same tokens twice."""
    from api.services import chat as chat_module

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(
        "src.tools.prompt_cache.gemini_context_cache", lambda *a, **k: "cachedContents/abc"
    )
    captured = {}

    def _fake_model(name, temperature=0.0, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(chat_module, "get_chat_model", _fake_model)
    _, prompt = chat_module._cached_model_and_prompt()
    assert prompt is None
    assert captured["cached_content"] == "cachedContents/abc"
