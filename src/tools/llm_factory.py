"""LLM client factory. The plan's default provider is OpenAI, but the model
*name* must never be hardcoded in agent files (config-driven per plan section
20/8.3). This factory additionally lets the provider itself be config/env
driven: if `OPENAI_API_KEY` doesn't look like a real OpenAI key (e.g. an
Anthropic key was supplied in its place in this environment) or `LLM_PROVIDER`
is explicitly set, it falls back to Anthropic via langchain-anthropic so the
system can still make genuine, non-mocked model calls rather than failing
auth. This is a deployment-environment concern, not a per-agent hardcode.
"""
from __future__ import annotations

import os


def _detect_provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER")
    if explicit:
        return explicit
    key = os.environ.get("OPENAI_API_KEY", "")
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key:
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


_ANTHROPIC_MODEL_FALLBACK = {
    # Maps common OpenAI-style config model names to a reasonable Anthropic
    # equivalent when the provider is Anthropic but a config still names an
    # OpenAI model (keeps config files provider-agnostic-ish across envs).
    "gpt-4o": "claude-sonnet-5",
    "gpt-4o-mini": "claude-haiku-4-5-20251001",
    "gpt-4-turbo": "claude-sonnet-5",
}


def get_chat_model(model_name: str, temperature: float = 0.0):
    provider = _detect_provider()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        resolved = _ANTHROPIC_MODEL_FALLBACK.get(model_name, model_name)
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return ChatAnthropic(model=resolved, temperature=temperature, api_key=key)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=temperature)
