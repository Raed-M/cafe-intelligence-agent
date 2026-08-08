from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import platform
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any


PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

MODEL_CATALOG: dict[str, list[dict[str, Any]]] = {
    "openai": [
        {
            "id": "gpt-5.6-sol",
            "name": "GPT-5.6 Sol",
            "summary": "Frontier reasoning for the hardest professional analysis.",
            "tier": "Frontier",
            "status": "Current",
            "input_price": 5.0,
            "cached_input_price": 0.5,
            "output_price": 30.0,
            "speed": "Most deliberate",
            "speed_rank": 1,
            "speed_note": "Expect the longest wait in this OpenAI set, especially at higher reasoning effort.",
            "context_window": "1.05M",
            "pricing_note": "Long inputs above 272K tokens use higher rates.",
            "recommended_for": "Quality-first weekly analysis",
        },
        {
            "id": "gpt-5.6-terra",
            "name": "GPT-5.6 Terra",
            "summary": "Strong current-generation reasoning with a lower price than Sol.",
            "tier": "Balanced intelligence",
            "status": "Current",
            "input_price": 2.5,
            "cached_input_price": 0.25,
            "output_price": 15.0,
            "speed": "Balanced",
            "speed_rank": 2,
            "speed_note": "A practical middle ground; reasoning effort still changes response time.",
            "context_window": "1.05M",
            "pricing_note": "Long inputs above 272K tokens use higher rates.",
            "recommended_for": "Balanced production use",
        },
        {
            "id": "gpt-5.6-luna",
            "name": "GPT-5.6 Luna",
            "summary": "The lowest-cost GPT-5.6 tier for frequent, focused work.",
            "tier": "Efficient reasoning",
            "status": "Current",
            "input_price": 1.0,
            "cached_input_price": 0.1,
            "output_price": 6.0,
            "speed": "Fast",
            "speed_rank": 3,
            "speed_note": "Usually quicker than Terra and Sol for the same reasoning setting.",
            "context_window": "1.05M",
            "pricing_note": "Long inputs above 272K tokens use higher rates.",
            "recommended_for": "Lower-cost summaries and extraction",
        },
        {
            "id": "gpt-4o-mini",
            "name": "GPT-4o mini",
            "summary": "Very low-cost previous-generation model for simple focused tasks.",
            "tier": "Ultra-low cost",
            "status": "Previous generation",
            "input_price": 0.15,
            "cached_input_price": 0.075,
            "output_price": 0.6,
            "speed": "Fastest",
            "speed_rank": 5,
            "speed_note": "Designed for low latency and high-volume focused requests.",
            "context_window": "128K",
            "pricing_note": None,
            "recommended_for": "Classification, extraction, and simple utility work",
        },
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "summary": "Previous-generation flagship retained for workflow compatibility.",
            "tier": "Compatibility",
            "status": "Previous generation",
            "input_price": 2.5,
            "cached_input_price": 1.25,
            "output_price": 10.0,
            "speed": "Fast",
            "speed_rank": 4,
            "speed_note": "Fast non-reasoning behavior, but not the lowest token price.",
            "context_window": "128K",
            "pricing_note": None,
            "recommended_for": "Existing verified workflow",
        },
    ],
    "anthropic": [
        {
            "id": "claude-fable-5",
            "name": "Claude Fable 5",
            "summary": "Anthropic's highest widely available capability for long-running agents.",
            "tier": "Highest capability",
            "status": "Current",
            "input_price": 10.0,
            "cached_input_price": None,
            "output_price": 50.0,
            "speed": "Slower",
            "speed_rank": 1,
            "speed_note": "Anthropic lists it as the slowest of these current Claude tiers.",
            "context_window": "1M",
            "pricing_note": None,
            "recommended_for": "The most demanding long-running analysis",
        },
        {
            "id": "claude-opus-5",
            "name": "Claude Opus 5",
            "summary": "Advanced model for complex agentic and enterprise workloads.",
            "tier": "Advanced",
            "status": "Current",
            "input_price": 5.0,
            "cached_input_price": None,
            "output_price": 25.0,
            "speed": "Moderate",
            "speed_rank": 2,
            "speed_note": "More responsive than Fable, with emphasis on complex work.",
            "context_window": "1M",
            "pricing_note": None,
            "recommended_for": "Complex multi-step analysis",
        },
        {
            "id": "claude-sonnet-5",
            "name": "Claude Sonnet 5",
            "summary": "Anthropic's strongest combination of speed and intelligence.",
            "tier": "Balanced intelligence",
            "status": "Current",
            "input_price": 2.0,
            "cached_input_price": None,
            "output_price": 10.0,
            "speed": "Fast",
            "speed_rank": 3,
            "speed_note": "The current balanced choice when turnaround matters.",
            "context_window": "1M",
            "pricing_note": "Intro price through Aug 31, 2026; standard price is $3 input / $15 output.",
            "recommended_for": "Balanced production analysis",
        },
        {
            "id": "claude-haiku-4-5-20251001",
            "name": "Claude Haiku 4.5",
            "summary": "The fastest Claude option here, with the smallest context window.",
            "tier": "Efficient",
            "status": "Current",
            "input_price": 1.0,
            "cached_input_price": None,
            "output_price": 5.0,
            "speed": "Fastest",
            "speed_rank": 4,
            "speed_note": "Anthropic's lowest-latency current tier in this comparison.",
            "context_window": "200K",
            "pricing_note": None,
            "recommended_for": "Fast structured analysis",
        }
    ],
    "gemini": [
        {
            "id": "gemini-3.1-pro-preview",
            "name": "Gemini 3.1 Pro",
            "summary": "Quality-first multimodal reasoning, currently offered as a preview model.",
            "tier": "Quality first",
            "status": "Preview",
            "input_price": 2.0,
            "cached_input_price": None,
            "output_price": 12.0,
            "speed": "Most deliberate",
            "speed_rank": 1,
            "speed_note": "Choose it for complexity rather than the shortest response time.",
            "context_window": "1M",
            "pricing_note": "Rates shown apply below 200K input tokens; longer prompts cost more.",
            "recommended_for": "Difficult reasoning workloads",
        },
        {
            "id": "gemini-3.6-flash",
            "name": "Gemini 3.6 Flash",
            "summary": "Google's current stable balance of speed and agentic intelligence.",
            "tier": "Balanced intelligence",
            "status": "Current",
            "input_price": 1.5,
            "cached_input_price": None,
            "output_price": 7.5,
            "speed": "Fast",
            "speed_rank": 2,
            "speed_note": "Built for quick multi-step and multimodal work.",
            "context_window": "1M",
            "pricing_note": None,
            "recommended_for": "Balanced production use",
        },
        {
            "id": "gemini-3.5-flash-lite",
            "name": "Gemini 3.5 Flash-Lite",
            "summary": "Google's fastest and lowest-cost current model for high throughput.",
            "tier": "Efficient",
            "status": "Current",
            "input_price": 0.3,
            "cached_input_price": None,
            "output_price": 2.5,
            "speed": "Fastest",
            "speed_rank": 3,
            "speed_note": "Optimized for extraction, routing, and large request volumes.",
            "context_window": "1M",
            "pricing_note": None,
            "recommended_for": "Extraction and frequent runs",
        },
        {
            # The model this project's .env actually ships with and which every
            # validated pipeline run to date used. It was missing from the
            # catalog, so the AI Connections page showed the provider selected
            # but no model highlighted.
            "id": "gemini-3.1-flash-lite",
            "name": "Gemini 3.1 Flash-Lite",
            "summary": "Previous-generation low-cost model; the project default for pipeline runs.",
            "tier": "Efficient",
            "status": "Previous generation",
            "input_price": 0.1,
            "cached_input_price": None,
            "output_price": 0.4,
            "speed": "Fastest",
            "speed_rank": 3,
            "speed_note": "Very high throughput on the free tier; used for the analyst/critic loop.",
            "context_window": "1M",
            "pricing_note": None,
            "recommended_for": "Low-cost analyst and critic runs",
        },
    ],
}

def _unlisted_model(model_id: str) -> dict[str, Any]:
    """Catalog card for a configured model the catalog does not know about.

    Prices are unknown rather than zero -- showing 0.0 would read as free. The
    UI renders `pricing_note` so the gap is explicit."""
    return {
        "id": model_id,
        "name": model_id,
        "summary": "Configured outside the built-in catalog; details unavailable.",
        "tier": "Unlisted",
        "status": "Current",
        "input_price": None,
        "cached_input_price": None,
        "output_price": None,
        "speed": "Unknown",
        "speed_rank": 99,
        "speed_note": "Not benchmarked here.",
        "context_window": "Unknown",
        "pricing_note": "Pricing not in the built-in catalog; check the provider's pricing page.",
        "recommended_for": "Currently configured model",
    }


DEFAULT_MODELS = {
    "openai": ("gpt-5.6-terra", "gpt-5.6-luna"),
    "anthropic": ("claude-sonnet-5", "claude-haiku-4-5-20251001"),
    "gemini": ("gemini-3.6-flash", "gemini-3.5-flash-lite"),
}

PROVIDER_SOURCES = {
    "openai": "https://developers.openai.com/api/docs/models/compare",
    "anthropic": "https://platform.claude.com/docs/en/about-claude/models/overview",
    "gemini": "https://ai.google.dev/gemini-api/docs/pricing",
}


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(value: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(value)
    return _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def _dpapi(value: bytes, *, decrypt: bool) -> bytes:
    if platform.system() != "Windows":
        raise RuntimeError("Persistent secret storage currently requires Windows DPAPI.")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    input_blob, input_buffer = _blob(value)
    entropy_blob, entropy_buffer = _blob(b"waddehha-ai-settings-v1")
    output_blob = _DataBlob()
    flags = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
    if decrypt:
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(input_blob), None, ctypes.byref(entropy_blob), None, None, flags, ctypes.byref(output_blob)
        )
    else:
        ok = crypt32.CryptProtectData(
            ctypes.byref(input_blob), "Waddehha AI settings", ctypes.byref(entropy_blob), None, None, flags,
            ctypes.byref(output_blob),
        )
    del input_buffer, entropy_buffer
    if not ok:
        raise OSError(ctypes.get_last_error(), "Windows DPAPI operation failed")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(output_blob.pbData, wintypes.HLOCAL))


class AiSettingsService:
    """Owner-managed runtime AI configuration whose secret values never leave the API."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._settings: dict[str, Any] = {}
        self._managed_env: set[str] = set()
        self._persisted = False
        self._load()

    @staticmethod
    def catalog(active_models: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
        """The model catalog, plus any model that is actually configured but
        not listed.

        A hardcoded catalog always lags real model releases, and when the
        configured model is missing from it the UI highlights nothing -- the
        page then looks unconfigured while the system is happily running that
        very model. Surfacing it as an "Unlisted" card keeps what is actually
        in use visible and selectable instead."""
        active_models = active_models or {}
        catalog: list[dict[str, Any]] = []
        for provider, models in MODEL_CATALOG.items():
            known = {model["id"] for model in models}
            extra = [
                _unlisted_model(model_id)
                for model_id in dict.fromkeys(active_models.get(provider, []))
                if model_id and model_id not in known
            ]
            catalog.append({
                "id": provider,
                "key_env": PROVIDER_KEY_ENV[provider],
                "source_url": PROVIDER_SOURCES[provider],
                "default_analysis_model": DEFAULT_MODELS[provider][0],
                "default_utility_model": DEFAULT_MODELS[provider][1],
                "models": [*models, *extra],
            })
        return catalog

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            encrypted = base64.b64decode(self.path.read_bytes(), validate=True)
            settings = json.loads(_dpapi(encrypted, decrypt=True).decode("utf-8"))
            if isinstance(settings, dict):
                self._settings = settings
                self._persisted = True
                self._apply_environment(settings)
        except Exception:
            # A corrupt or another-user DPAPI blob must never prevent the app from starting.
            self._settings = {}
            self._persisted = False

    @staticmethod
    def _fingerprint(value: str | None) -> str | None:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10] if value else None

    def public(self) -> dict[str, Any]:
        provider = str(self._settings.get("provider") or os.getenv("LLM_PROVIDER", "")).lower()
        if provider not in PROVIDER_KEY_ENV:
            provider = ""
        provider_key = self._settings.get("api_key") or (os.getenv(PROVIDER_KEY_ENV.get(provider, "")) if provider else None)
        primary_default, utility_default = DEFAULT_MODELS.get(provider, ("", ""))
        tavily = self._settings.get("tavily_key") or os.getenv("TAVILY_API_KEY")
        langsmith = self._settings.get("langsmith_key") or os.getenv("LANGCHAIN_API_KEY")
        analysis_model = self._settings.get("analysis_model") or os.getenv("ANALYST_MODEL") or primary_default
        utility_model = self._settings.get("utility_model") or os.getenv("EMAIL_EXTRACTOR_MODEL") or utility_default
        return {
            "provider": provider or None,
            "provider_configured": bool(provider_key),
            "provider_fingerprint": self._fingerprint(provider_key),
            "analysis_model": analysis_model,
            "utility_model": utility_model,
            "tavily_configured": bool(tavily),
            "tavily_fingerprint": self._fingerprint(tavily),
            "langsmith_configured": bool(langsmith),
            "langsmith_fingerprint": self._fingerprint(langsmith),
            "persisted": self._persisted,
            "persistence_available": platform.system() == "Windows",
            "catalog": self.catalog(
                {provider: [analysis_model, utility_model]} if provider else None
            ),
        }

    def _apply_environment(self, settings: dict[str, Any]) -> None:
        provider = settings["provider"]
        primary = settings["analysis_model"]
        utility = settings.get("utility_model") or primary
        values = {
            "LLM_PROVIDER": provider,
            PROVIDER_KEY_ENV[provider]: settings["api_key"],
            "ANALYST_MODEL": primary,
            "CRITIC_MODEL": primary,
            "CONTENT_MODEL": primary,
            "CONTENT_VALIDATOR_MODEL": primary,
            "REPORT_SUMMARY_MODEL": utility,
            "EMAIL_EXTRACTOR_MODEL": utility,
        }
        if settings.get("tavily_key"):
            values["TAVILY_API_KEY"] = settings["tavily_key"]
        if settings.get("langsmith_key"):
            values["LANGCHAIN_API_KEY"] = settings["langsmith_key"]
            values["LANGCHAIN_TRACING_V2"] = "true"
        for key, value in values.items():
            os.environ[key] = str(value)
            self._managed_env.add(key)

    def inject_environment(self) -> None:
        with self._lock:
            if self._settings:
                self._apply_environment(self._settings)

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = dict(self._settings)
            provider = payload["provider"]
            allowed_models = {model["id"] for model in MODEL_CATALOG[provider]}
            analysis_model = payload["analysis_model"]
            utility_model = payload.get("utility_model") or analysis_model
            if analysis_model not in allowed_models or utility_model not in allowed_models:
                raise ValueError("The selected model is not compatible with this provider.")
            api_key = payload.get("api_key") or (
                current.get("api_key") if current.get("provider") == provider else None
            ) or os.getenv(PROVIDER_KEY_ENV[provider])
            if not api_key:
                raise ValueError("An API key is required for the selected provider.")
            settings = {
                "provider": provider,
                "api_key": api_key,
                "analysis_model": analysis_model,
                "utility_model": utility_model,
                "tavily_key": payload.get("tavily_key") or current.get("tavily_key") or os.getenv("TAVILY_API_KEY"),
                "langsmith_key": payload.get("langsmith_key") or current.get("langsmith_key") or os.getenv("LANGCHAIN_API_KEY"),
            }
            self._settings = settings
            self._apply_environment(settings)
            remember = bool(payload.get("remember"))
            if remember:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                protected = _dpapi(json.dumps(settings, separators=(",", ":")).encode("utf-8"), decrypt=False)
                self.path.write_bytes(base64.b64encode(protected))
                self._persisted = True
            else:
                self.path.unlink(missing_ok=True)
                self._persisted = False
            return self.public()

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self.path.unlink(missing_ok=True)
            self._settings = {}
            self._persisted = False
            for key in self._managed_env:
                os.environ.pop(key, None)
            self._managed_env.clear()
            return self.public()

    def test_connection(self) -> dict[str, Any]:
        current = self.public()
        if not current["provider_configured"] or not current["analysis_model"]:
            raise ValueError("Save a provider, API key, and analysis model first.")
        from src.tools.llm_factory import get_chat_model

        started = time.perf_counter()
        try:
            model = get_chat_model(current["analysis_model"], temperature=0)
            model.invoke("Return exactly the word OK. Do not use tools.")
        except Exception as exc:
            return {
                "ok": False,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "message": f"The provider rejected the test ({type(exc).__name__}). Check the key and model access.",
            }
        return {
            "ok": True,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "message": "Connection verified. No cafe data was included in this test.",
        }
