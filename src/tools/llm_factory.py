"""LLM client factory. Model *names* are always config-driven (plan section
20/8.3, never hardcoded in agent files) and now so is the *provider*: set
LLM_PROVIDER=anthropic|openai|gemini (default anthropic) plus that provider's
API key, and every agent in config/app_settings.yaml (analyst/critic/content/
content_validator/report_summary/email_extractor) switches together -- no
per-agent code changes, no key-format sniffing.
"""
from __future__ import annotations

import os

_PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def get_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()


def provider_env_var(provider: str | None = None) -> str | None:
    return _PROVIDER_ENV_KEYS.get(provider or get_provider())


def has_provider_key(provider: str | None = None) -> bool:
    env_key = provider_env_var(provider)
    return bool(env_key and os.environ.get(env_key))


def get_chat_model(model_name: str, temperature: float = 0.0):
    provider = get_provider()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; supported: {sorted(_PROVIDER_ENV_KEYS)}")
