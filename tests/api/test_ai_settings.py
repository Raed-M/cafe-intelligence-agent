from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.services.ai_settings import AiSettingsService


MANAGED_ENV = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "ANALYST_MODEL",
    "CRITIC_MODEL",
    "CONTENT_MODEL",
    "CONTENT_VALIDATOR_MODEL",
    "REPORT_SUMMARY_MODEL",
    "EMAIL_EXTRACTOR_MODEL",
    "TAVILY_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
)


def test_ai_settings_are_owner_only_and_never_return_secrets(
    app, client: TestClient, login, tmp_path: Path
) -> None:
    secret = "sk-test-secret-that-must-never-be-returned"
    tavily = "tvly-test-secret-that-must-never-be-returned"
    previous = {name: os.environ.get(name) for name in MANAGED_ENV}
    app.state.ai_settings.path = tmp_path / "ai-settings.dpapi"

    try:
        assert client.get("/api/admin/ai-settings").status_code == 401
        login("manager")
        assert client.get("/api/admin/ai-settings").status_code == 403
        client.post("/api/auth/logout")
        login("owner")

        saved = client.post(
            "/api/admin/ai-settings",
            json={
                "provider": "openai",
                "api_key": secret,
                "analysis_model": "gpt-5.6-terra",
                "utility_model": "gpt-5.6-luna",
                "tavily_key": tavily,
                "remember": False,
            },
        )
        assert saved.status_code == 200
        assert saved.headers["cache-control"] == "no-store, max-age=0"
        assert secret not in saved.text
        assert tavily not in saved.text

        settings = saved.json()["settings"]
        assert settings["provider"] == "openai"
        assert settings["provider_configured"] is True
        assert settings["provider_fingerprint"] == hashlib.sha256(secret.encode()).hexdigest()[:10]
        assert settings["tavily_configured"] is True
        assert settings["persisted"] is False
        assert not app.state.ai_settings.path.exists()
        assert os.environ["OPENAI_API_KEY"] == secret
        assert os.environ["ANALYST_MODEL"] == "gpt-5.6-terra"
        assert os.environ["EMAIL_EXTRACTOR_MODEL"] == "gpt-5.6-luna"

        fetched = client.get("/api/admin/ai-settings")
        assert fetched.status_code == 200
        assert secret not in fetched.text
        assert tavily not in fetched.text
        openai = next(item for item in fetched.json()["settings"]["catalog"] if item["id"] == "openai")
        mini = next(model for model in openai["models"] if model["id"] == "gpt-4o-mini")
        assert mini["input_price"] == 0.15
        assert mini["output_price"] == 0.6
        assert mini["speed"] == "Fastest"
        assert "quality" not in mini
        assert "latency" not in mini

        cleared = client.delete("/api/admin/ai-settings")
        assert cleared.status_code == 200
        assert cleared.json()["settings"]["provider_configured"] is False
        assert secret not in cleared.text
    finally:
        app.state.ai_settings.clear()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_ai_settings_reject_a_model_from_another_provider(
    app, client: TestClient, login, tmp_path: Path
) -> None:
    app.state.ai_settings.path = tmp_path / "ai-settings.dpapi"
    login("owner")
    response = client.post(
        "/api/admin/ai-settings",
        json={
            "provider": "gemini",
            "api_key": "fake-test-key",
            "analysis_model": "gpt-5.6-sol",
            "remember": False,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "ai_settings_invalid"
    assert "fake-test-key" not in response.text


def test_remembered_settings_use_windows_account_encryption(tmp_path: Path) -> None:
    secret = "sk-local-dpapi-verification-secret"
    path = tmp_path / "ai-settings.dpapi"
    service = AiSettingsService(path)
    try:
        saved = service.save(
            {
                "provider": "openai",
                "api_key": secret,
                "analysis_model": "gpt-5.6-terra",
                "utility_model": "gpt-5.6-luna",
                "remember": True,
            }
        )
        assert saved["persisted"] is True
        assert path.exists()
        assert secret.encode() not in path.read_bytes()

        restored = AiSettingsService(path)
        public = restored.public()
        assert public["provider"] == "openai"
        assert public["provider_configured"] is True
        assert public["provider_fingerprint"] == hashlib.sha256(secret.encode()).hexdigest()[:10]
    finally:
        service.clear()
