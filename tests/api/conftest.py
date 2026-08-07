from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.config import ApiSettings
from api.main import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSWORDS = {
    "owner": "owner-password-1234",
    "manager": "manager-password-1234",
    "employee": "employee-password-1234",
}


@pytest.fixture
def app(tmp_path: Path):
    accounts = tuple(
        {
            "role": role,
            "email": f"{role}@test.local",
            "password": password,
            "display_name": role.title(),
        }
        for role, password in PASSWORDS.items()
    )
    settings = ApiSettings(
        project_root=PROJECT_ROOT,
        api_db=tmp_path / "api.sqlite3",
        environment="development",
        cookie_secure=False,
        seed_users=True,
        seed_accounts=accounts,
        include_test_evidence=True,
        runs_enabled=False,
    )
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def login(client: TestClient):
    def do_login(role: str) -> dict:
        response = client.post(
            "/api/auth/login",
            json={"email": f"{role}@test.local", "password": PASSWORDS[role]},
        )
        assert response.status_code == 200
        return response.json()["user"]

    return do_login
