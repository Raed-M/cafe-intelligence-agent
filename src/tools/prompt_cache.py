"""Provider-aware prompt caching for long, static system prompts.

Why this is not one mechanism
-----------------------------
The three providers this project supports cache prompts in three different ways,
so "enable prompt caching" is necessarily provider-specific:

* **Anthropic** -- explicit and opt-in. A `cache_control: {"type": "ephemeral"}`
  marker on a system content block tells the API to cache that prefix. Nothing
  is cached without the marker, so this is the one provider where code is
  strictly required to get any benefit.
* **OpenAI** -- automatic. Prompts above the provider's minimum are cached with
  no API surface at all; the only requirement is that the prefix stays
  byte-identical between calls, which is a prompt-construction concern rather
  than a parameter.
* **Gemini** -- two tiers. *Implicit* caching is automatic on eligible models
  above a minimum prefix size; *explicit* caching requires creating a
  CachedContent object and passing its name as `cached_content`.

Measured on this project (2026-08-08, gemini-3.1-flash-lite, 1,034-token system
prompt): implicit caching reported `cache_read=0` on three identical-prefix
calls, and explicit cache creation is refused outright on the free tier --
`TotalCachedContentStorageTokensPerModelFreeTier ... limit=0`. So on the current
key this module is inert by design: it degrades silently rather than failing,
and starts paying off without code changes on a paid Gemini tier or after
switching provider.

Cache hits are visible in the telemetry `cache_read_tokens` field
(LLM_TELEMETRY_PATH), which is how the claims above were established and how any
future improvement should be confirmed rather than assumed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

GEMINI_CACHE_TTL_SECONDS = int(os.environ.get("LLM_PROMPT_CACHE_TTL_SECONDS", "1800"))
"""How long a Gemini explicit cache is kept alive. Long enough that a burst of
chat turns shares one cache, short enough that storage (which is billed) is not
held indefinitely."""


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def caching_enabled() -> bool:
    """LLM_PROMPT_CACHE=0 disables every provider path here.

    Default on: the always-on paths cost nothing. Anthropic markers are inert if
    the prefix is too short, and OpenAI caching needs no code at all."""
    return _flag("LLM_PROMPT_CACHE", "1")


def gemini_explicit_caching_enabled() -> bool:
    """LLM_PROMPT_CACHE_GEMINI_EXPLICIT=1 opts into Gemini explicit caching.

    Off by default, unlike the other providers, because creating a Gemini cache
    costs a real HTTP round-trip that *cannot* succeed on the free tier -- the
    storage quota there is literally zero
    (`TotalCachedContentStorageTokensPerModelFreeTier ... limit=0`). Attempting
    it by default bought one doomed request per process and, worse, pulled the
    otherwise-offline test suite onto the network.

    Turn this on when running against a paid tier; nothing else needs to
    change, and cache hits will show up as telemetry `cache_read_tokens`."""
    return caching_enabled() and _flag("LLM_PROMPT_CACHE_GEMINI_EXPLICIT", "0")


def cacheable_system_prompt(text: str, provider: str) -> Any:
    """The system prompt in the most cacheable form for `provider`.

    Anthropic gets a SystemMessage whose content is a cache_control-marked text
    block; every other provider gets the plain string (OpenAI caches
    automatically, Gemini via `cached_content` -- handled separately).

    The Anthropic form is wrapped in a SystemMessage rather than returned as a
    bare block list because that is what consumers accept: create_react_agent
    takes `str | SystemMessage | callable`, and handing it a raw list silently
    changed how the prompt was applied instead of failing loudly."""
    if not caching_enabled() or provider != "anthropic":
        return text
    from langchain_core.messages import SystemMessage

    return SystemMessage(
        content=[{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    )


# --- Gemini explicit context cache -----------------------------------------

_gemini_cache_lock = threading.Lock()
_gemini_caches: dict[str, tuple[str, float]] = {}
"""key -> (cache resource name, unix expiry). Keyed by model + prompt hash so a
changed prompt never reuses a stale cache."""

_gemini_unavailable: set[str] = set()
"""Models whose explicit cache creation was refused. Remembered so a free-tier
key does not re-attempt (and re-log) on every single request."""


def _cache_key(model_name: str, system_prompt: str) -> str:
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    return f"{model_name}:{digest}"


def gemini_context_cache(model_name: str, system_prompt: str) -> str | None:
    """Resource name of a Gemini explicit cache holding `system_prompt`, or None.

    Returns None -- never raises -- when caching is disabled, the SDK is absent,
    or the provider refuses (free tier, prompt below the minimum cacheable
    size). Callers must treat None as "send the prompt normally".

    IMPORTANT for callers: when this returns a name, the system prompt is *in*
    the cache and must NOT also be sent with the request, or it is billed twice.
    """
    if not gemini_explicit_caching_enabled():
        return None
    key = _cache_key(model_name, system_prompt)
    now = time.time()

    with _gemini_cache_lock:
        if key in _gemini_unavailable:
            return None
        cached = _gemini_caches.get(key)
        if cached and cached[1] > now + 30:  # keep a margin against expiry mid-request
            return cached[0]

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        with _gemini_cache_lock:
            _gemini_unavailable.add(key)
        return None

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                system_instruction=system_prompt,
                ttl=f"{GEMINI_CACHE_TTL_SECONDS}s",
                display_name="cafe-intelligence-system-prompt",
            ),
        )
    except Exception as e:  # noqa: BLE001
        # Expected on the free tier: "TotalCachedContentStorageTokensPerModel
        # FreeTier limit exceeded ... limit=0". Also covers a prompt below the
        # model's minimum cacheable size. Either way: no cache, no failure.
        with _gemini_cache_lock:
            _gemini_unavailable.add(key)
        logger.info(
            "Gemini prompt caching unavailable for %s; sending the prompt uncached. Reason: %s",
            model_name, str(e)[:200],
        )
        return None

    with _gemini_cache_lock:
        _gemini_caches[key] = (cache.name, now + GEMINI_CACHE_TTL_SECONDS)
    logger.info("Created Gemini prompt cache %s for %s", cache.name, model_name)
    return cache.name


def reset_for_tests() -> None:
    """Clears memoized cache handles and unavailability marks."""
    with _gemini_cache_lock:
        _gemini_caches.clear()
        _gemini_unavailable.clear()
