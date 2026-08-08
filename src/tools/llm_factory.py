"""LLM client factory. Model *names* are always config-driven (plan section
20/8.3, never hardcoded in agent files) and now so is the *provider*: set
LLM_PROVIDER=anthropic|openai|gemini (default anthropic) plus that provider's
API key, and every agent in config/app_settings.yaml (analyst/critic/content/
content_validator/report_summary/email_extractor) switches together -- no
per-agent code changes, no key-format sniffing.

LLM_RPM_LIMIT (optional): when set to a positive integer, every call through
get_chat_model() -- including .with_structured_output(...).invoke(...) -- is
throttled to at most that many requests per rolling 60s window, process-wide.
Meant for free/low-tier providers with a hard RPM cap (e.g. Gemini's 10-15
RPM free tier); leave unset for providers with generous limits.

{PROVIDER}_API_KEYS (optional, e.g. GOOGLE_API_KEYS): comma-separated list of
keys to rotate through when one hits a quota-exhausted (429/RESOURCE_EXHAUSTED)
error -- e.g. Gemini's free tier also caps *daily* requests per key (500/day),
separately from the RPM limit above; multiple keys multiply that daily
allowance without buying a paid tier. Falls back to the single {PROVIDER}_API_KEY
when unset. Rotation only triggers on a quota error, never spends an extra
request on a healthy key.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

_PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

# --- opt-in local telemetry -------------------------------------------------
# LLM_TELEMETRY_PATH (optional): when set, append one JSON line per provider
# request to that file -- model, graph node, latency, token usage, error type.
# Implemented as a callback handler rather than by wrapping .invoke() because
# nearly every call here goes through .with_structured_output(), whose return
# value is a parsed pydantic object with no usage_metadata on it; the callback
# still sees the underlying AIMessage and therefore the real token counts.
# Default off, so normal runs and the test suite are completely unaffected.


class _TelemetryHandler(BaseCallbackHandler):
    """Writes one JSONL record per LLM request. Deliberately swallows its own
    errors: telemetry must never be able to fail a run."""

    raise_error = False

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._started: dict[Any, dict[str, Any]] = {}

    def _write(self, record: dict[str, Any]) -> None:
        try:
            import json

            with self._lock, open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _begin(self, run_id, metadata, prompts_chars):
        self._started[run_id] = {
            "t0": time.time(),
            "node": (metadata or {}).get("langgraph_node"),
            "step": (metadata or {}).get("langgraph_step"),
            "prompt_chars": prompts_chars,
        }

    def on_llm_start(self, serialized, prompts, *, run_id=None, metadata=None, **kwargs):
        self._begin(run_id, metadata, sum(len(p) for p in prompts or []))

    def on_chat_model_start(self, serialized, messages, *, run_id=None, metadata=None, **kwargs):
        chars = sum(len(str(getattr(m, "content", m))) for batch in messages or [] for m in batch)
        self._begin(run_id, metadata, chars)

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        started = self._started.pop(run_id, {})
        usage, model, finish = {}, None, None
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            usage = getattr(msg, "usage_metadata", None) or {}
            meta = getattr(msg, "response_metadata", {}) or {}
            model = meta.get("model_name") or meta.get("model")
            finish = getattr(gen, "generation_info", {}) or {}
            finish = finish.get("finish_reason")
        except Exception:  # noqa: BLE001
            pass
        # Cache accounting: providers report cache hits under
        # usage_metadata["input_token_details"] with provider-specific keys
        # (cache_read for Gemini/OpenAI, cache_read/cache_creation for
        # Anthropic). Recorded so prompt-caching work can be measured instead
        # of assumed.
        details = usage.get("input_token_details") or {}
        self._write({
            "ts": time.time(), "status": "ok", "node": started.get("node"),
            "step": started.get("step"), "model": model,
            "latency_s": round(time.time() - started.get("t0", time.time()), 3),
            "prompt_chars": started.get("prompt_chars"),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"), "finish_reason": finish,
            "cache_read_tokens": details.get("cache_read"),
            "cache_creation_tokens": details.get("cache_creation"),
        })

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        started = self._started.pop(run_id, {})
        msg = str(error)
        self._write({
            "ts": time.time(), "status": "error", "node": started.get("node"),
            "step": started.get("step"),
            "latency_s": round(time.time() - started.get("t0", time.time()), 3),
            "prompt_chars": started.get("prompt_chars"),
            "error_type": type(error).__name__,
            "quota_exhausted": _is_quota_exhausted(error),
            "error": msg[:500],
        })


_telemetry_handler: _TelemetryHandler | None = None
_telemetry_path: str | None = None


def _get_telemetry_callbacks() -> list[Any]:
    global _telemetry_handler, _telemetry_path
    path = os.environ.get("LLM_TELEMETRY_PATH", "").strip()
    if not path:
        return []
    if path != _telemetry_path:
        _telemetry_handler = _TelemetryHandler(path)
        _telemetry_path = path
    return [_telemetry_handler]


def get_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()


def provider_env_var(provider: str | None = None) -> str | None:
    return _PROVIDER_ENV_KEYS.get(provider or get_provider())


def has_provider_key(provider: str | None = None) -> bool:
    env_key = provider_env_var(provider)
    return bool(env_key and os.environ.get(env_key))


def _key_pool(provider: str) -> list[str]:
    env_key = provider_env_var(provider)
    if not env_key:
        return []
    plural = os.environ.get(f"{env_key}S", "")  # e.g. GOOGLE_API_KEY -> GOOGLE_API_KEYS
    keys = [k.strip() for k in plural.split(",") if k.strip()]
    if keys:
        return keys
    single = os.environ.get(env_key, "")
    return [single] if single else []


def _is_quota_exhausted(exc: Exception) -> bool:
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg or "quota" in msg.lower()


_QUOTA_COOLDOWN_SECONDS = int(os.environ.get("LLM_QUOTA_COOLDOWN_SECONDS", "1800"))
_quota_exhausted_until: dict[str, float] = {}


def _circuit_open(provider: str) -> float | None:
    until = _quota_exhausted_until.get(provider)
    return until if until and time.monotonic() < until else None


def _trip_circuit(provider: str) -> None:
    _quota_exhausted_until[provider] = time.monotonic() + _QUOTA_COOLDOWN_SECONDS


class _CircuitBreakerChatModel:
    """Wraps a chat model (rotating or not) so that once its provider's quota
    has been confirmed exhausted, every subsequent call fails instantly --
    zero network calls -- for a cooldown window, instead of re-spending a
    doomed request. This matters specifically because callers we don't
    control can retry automatically: `langgraph dev`/LangGraph Platform
    restarts a run from its last checkpoint after a worker crash, which
    re-issues every in-flight LLM call from scratch. Without this, that kind
    of external auto-retry (or simple heavy concurrent use) turns one
    exhausted-quota error into an unbounded loop of doomed calls. Only trips
    on a quota-exhausted error -- every other error passes through untouched
    so genuine transient failures are still retried normally by callers."""

    def __init__(self, inner: Any, provider: str):
        self._inner = inner
        self._provider = provider

    def invoke(self, *args, **kwargs):
        open_until = _circuit_open(self._provider)
        if open_until is not None:
            raise RuntimeError(
                f"{self._provider} quota circuit breaker open until "
                f"{time.strftime('%H:%M:%S', time.localtime(open_until))} -- its quota was confirmed "
                f"exhausted within the last {_QUOTA_COOLDOWN_SECONDS}s; refusing to spend another call "
                f"until the cooldown passes (set LLM_QUOTA_COOLDOWN_SECONDS to change this window)."
            )
        try:
            return self._inner.invoke(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if _is_quota_exhausted(e):
                _trip_circuit(self._provider)
            raise

    def with_structured_output(self, *args, **kwargs):
        return _CircuitBreakerChatModel(self._inner.with_structured_output(*args, **kwargs), self._provider)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _RotatingKeyChatModel:
    """Holds one chat-model instance per API key and retries the *same*
    request on the next key when the current one reports quota exhaustion.
    Never retries on any other error -- those propagate immediately."""

    def __init__(self, build_fn, keys: list[str], start_index: int = 0):
        self._build_fn = build_fn
        self._keys = keys
        self._index = start_index
        self._models: list[Any] = [None] * len(keys)

    def _current(self):
        if self._models[self._index] is None:
            self._models[self._index] = self._build_fn(self._keys[self._index])
        return self._models[self._index]

    def invoke(self, *args, **kwargs):
        last_exc: Exception | None = None
        for _ in range(len(self._keys)):
            try:
                return self._current().invoke(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if not _is_quota_exhausted(e):
                    raise
                last_exc = e
                self._index = (self._index + 1) % len(self._keys)
        raise last_exc  # type: ignore[misc]

    def with_structured_output(self, *args, **kwargs):
        return _RotatingKeyChatModel(
            lambda key: self._build_fn(key).with_structured_output(*args, **kwargs),
            self._keys, self._index,
        )

    def __getattr__(self, name):
        return getattr(self._current(), name)


class _RateLimiter:
    """Thread-safe sliding-window limiter: blocks in acquire() until fewer
    than max_calls timestamps remain within the trailing `period` seconds."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0]) + 0.05
            time.sleep(max(wait, 0.05))


_rate_limiter: _RateLimiter | None = None
_rate_limiter_limit: int | None = None


def _get_rate_limiter() -> _RateLimiter | None:
    global _rate_limiter, _rate_limiter_limit
    raw = os.environ.get("LLM_RPM_LIMIT", "").strip()
    limit = int(raw) if raw.isdigit() and int(raw) > 0 else None
    if limit != _rate_limiter_limit:
        _rate_limiter = _RateLimiter(limit) if limit else None
        _rate_limiter_limit = limit
    return _rate_limiter


class _RateLimitedChatModel:
    """Wraps a langchain chat model (or a .with_structured_output(...) result)
    so every .invoke() call goes through the shared rate limiter first. Any
    other attribute/method (bind_tools, etc.) passes through unwrapped."""

    def __init__(self, inner, limiter: _RateLimiter):
        self._inner = inner
        self._limiter = limiter

    def invoke(self, *args, **kwargs):
        self._limiter.acquire()
        return self._inner.invoke(*args, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        return _RateLimitedChatModel(self._inner.with_structured_output(*args, **kwargs), self._limiter)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def extract_text(response) -> str:
    """Normalizes a chat model response's .content across providers. Anthropic
    and OpenAI return a plain string; Gemini returns a list of content-block
    dicts (e.g. [{"type": "text", "text": "...", "extras": {...}}]) -- naively
    str()-ing that list corrupts any downstream regex/JSON parsing, so pull
    the text out of each block instead."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        if parts:
            return "".join(parts)
    return str(content)


def get_chat_model(model_name: str, temperature: float = 0.0, **provider_kwargs):
    """Builds a chat model for the active provider.

    provider_kwargs are passed straight through to the underlying client, for
    options only one provider has -- e.g. Gemini's `cached_content` (an explicit
    prompt cache; see src/tools/prompt_cache.py). Passing a kwarg the active
    provider does not accept is a programming error and will raise, so callers
    should send provider-specific options only when that provider is active.
    """
    provider = get_provider()
    cb = _get_telemetry_callbacks()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model=model_name, temperature=temperature, callbacks=cb, **provider_kwargs)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=model_name, temperature=temperature, callbacks=cb, **provider_kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # max_retries=1 (i.e. no SDK-internal retry): ChatGoogleGenerativeAI
        # defaults to max_retries=6, which retries-with-backoff *inside* a
        # single .invoke() call before ever raising to our code -- on a
        # daily-quota 429 (not transient), that just burns minutes retrying
        # the same doomed key instead of giving _RotatingKeyChatModel a
        # chance to switch keys immediately. The library's own docs recommend
        # exactly this (max_retries=1 + custom retry logic) for callers that
        # implement their own retry/rotation, which is what we do below.
        keys = _key_pool("gemini")
        if len(keys) > 1:
            model = _RotatingKeyChatModel(
                lambda key: ChatGoogleGenerativeAI(
                    model=model_name, temperature=temperature, google_api_key=key,
                    max_retries=1, callbacks=cb, **provider_kwargs,
                ),
                keys,
            )
        else:
            model = ChatGoogleGenerativeAI(
                model=model_name, temperature=temperature, max_retries=1, callbacks=cb,
                **provider_kwargs,
            )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; supported: {sorted(_PROVIDER_ENV_KEYS)}")

    limiter = _get_rate_limiter()
    result = _RateLimitedChatModel(model, limiter) if limiter else model
    if provider == "gemini":
        # Outermost wrapper: check the circuit breaker before even acquiring a
        # rate-limit slot, so a tripped breaker fails instantly instead of
        # waiting out an RPM sleep first just to fail anyway.
        result = _CircuitBreakerChatModel(result, "gemini")
    return result
