from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.config import ApiSettings
from api.main import create_app


def test_signup_stays_pending_until_owner_approves(client: TestClient) -> None:
    signup = client.post(
        "/api/auth/signup",
        json={
            "display_name": "New Manager",
            "email": "new.manager@example.com",
            "password": "manager-pass-123",
            "requested_role": "manager",
        },
    )
    assert signup.status_code == 202
    request_record = signup.json()["access_request"]
    assert request_record["status"] == "pending"

    pending_login = client.post(
        "/api/auth/login",
        json={"email": "new.manager@example.com", "password": "manager-pass-123"},
    )
    assert pending_login.status_code == 403
    assert pending_login.json()["error"]["code"] == "approval_pending"

    admin_login = client.post("/api/auth/login", json={"email": "admin", "password": "admin"})
    assert admin_login.status_code == 200
    assert admin_login.json()["user"]["role"] == "owner"
    pending = client.get("/api/admin/access-requests?status=pending")
    assert pending.status_code == 200
    assert request_record["id"] in {item["id"] for item in pending.json()["items"]}
    approved = client.post(
        f"/api/admin/access-requests/{request_record['id']}/decision",
        json={"decision": "approve"},
    )
    assert approved.status_code == 200
    assert approved.json()["access_request"]["status"] == "approved"

    client.post("/api/auth/logout")
    accepted_login = client.post(
        "/api/auth/login",
        json={"email": "new.manager@example.com", "password": "manager-pass-123"},
    )
    assert accepted_login.status_code == 200
    assert accepted_login.json()["user"]["role"] == "manager"


def test_rejected_signup_cannot_sign_in(client: TestClient) -> None:
    request_record = client.post(
        "/api/auth/signup",
        json={
            "display_name": "Rejected Employee",
            "email": "rejected@example.com",
            "password": "employee-pass-123",
            "requested_role": "employee",
        },
    ).json()["access_request"]
    assert client.post("/api/auth/login", json={"email": "admin", "password": "admin"}).status_code == 200
    assert client.post(
        f"/api/admin/access-requests/{request_record['id']}/decision",
        json={"decision": "reject"},
    ).status_code == 200
    client.post("/api/auth/logout")
    rejected = client.post(
        "/api/auth/login",
        json={"email": "rejected@example.com", "password": "employee-pass-123"},
    )
    assert rejected.status_code == 403
    assert rejected.json()["error"]["code"] == "access_rejected"


def test_browser_upload_processes_registered_file_and_starts_run(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "test-cafe"
    config_dir = tmp_path / "config"
    data_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    (data_dir / "cafe_profile.json").write_text(
        json.dumps(
            {
                "cafe_name": "Upload Test Café",
                "city": "Riyadh",
                "region": "Riyadh",
                "country": "Saudi Arabia",
                "timezone": "Asia/Riyadh",
                "currency": "SAR",
                "seats": 20,
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "source_registry.yaml").write_text(
        "sources:\n  - name: pos\n    parser: parse_pos\n    path: pos_transactions.csv\n    required_for: [sales]\n",
        encoding="utf-8",
    )
    original = b"transaction_id,amount\nold,1\n"
    replacement = b"transaction_id,amount\nnew,2\n"
    (data_dir / "pos_transactions.csv").write_bytes(original)

    settings = ApiSettings(
        project_root=tmp_path,
        api_db=tmp_path / "db" / "api.sqlite3",
        environment="development",
        cookie_secure=False,
        bootstrap_admin=True,
        runs_enabled=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.post("/api/auth/login", json={"email": "admin", "password": "admin"}).status_code == 200
        cafe_id = client.get("/api/cafes").json()["items"][0]["id"]
        processed = client.post(
            f"/api/cafes/{cafe_id}/data/process",
            json={
                "files": [
                    {
                        "name": "pos_transactions.csv",
                        "relative_path": "selected-folder/pos_transactions.csv",
                        "media_type": "text/csv",
                        "size": len(replacement),
                        "last_modified": "2026-08-06T10:00:00.000Z",
                        "content_base64": base64.b64encode(replacement).decode(),
                    }
                ]
            },
        )
        assert processed.status_code == 202
        payload = processed.json()
        assert payload["upload"]["accepted"][0]["source"] == "pos"
        assert payload["upload"]["accepted"][0]["replaced_existing"] is True
        assert payload["run"]["status"] == "failed"
        assert payload["run"]["stage"] == "disabled"
        assert (data_dir / "pos_transactions.csv").read_bytes() == replacement
        backups = list((tmp_path / "outputs" / "uploads").glob("**/backup/pos_transactions.csv"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == original
