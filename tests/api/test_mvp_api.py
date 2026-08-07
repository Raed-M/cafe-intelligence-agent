from __future__ import annotations

import json
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient

from api.artifacts import ArtifactRepository, sanitize_report_html, strip_path_text
from api.config import ApiSettings
from api.database import ApiDatabase
from api.services.chat import grounded_answer


PATH_LEAK = re.compile(
    r"(?:file://|[A-Za-z]:[\\/]|\\\\|/(?:home|users|tmp|var|opt|workspace|mnt)/|(?:outputs|data|db)[\\/])",
    re.IGNORECASE,
)


def assert_no_path_leak(payload) -> None:
    if isinstance(payload, str):
        assert PATH_LEAK.search(payload) is None
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert_no_path_leak(key)
            assert_no_path_leak(value)
        return
    if isinstance(payload, (list, tuple, set)):
        for value in payload:
            assert_no_path_leak(value)


def test_health_and_structured_auth_error(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {
        "schema_version": "1.0",
        "status": "ok",
        "service": "waddehha-api",
        "version": "0.1.0",
    }

    unauthorized = client.get("/api/auth/me")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "authentication_required"
    assert unauthorized.json()["error"]["request_id"].startswith("req-")


def test_cookie_login_me_logout_and_no_secret_exposure(client: TestClient, login) -> None:
    response = client.post(
        "/api/auth/login",
        json={"email": "owner@test.local", "password": "owner-password-1234"},
    )
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/api" in cookie
    assert "token" not in response.json()
    assert "password" not in json.dumps(response.json()).lower()
    assert client.get("/api/auth/me").json()["user"]["role"] == "owner"

    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401
    assert ApiSettings(
        project_root=Path.cwd(), environment="production", cookie_secure=False
    ).cookie_secure is True


def test_seed_accounts_are_development_only(tmp_path: Path) -> None:
    try:
        ApiSettings(
            project_root=tmp_path,
            api_db=tmp_path / "api.sqlite",
            environment="production",
            seed_users=True,
            seed_accounts=(),
        )
    except RuntimeError as exc:
        assert "outside development" in str(exc)
    else:
        raise AssertionError("production seed users must be rejected")
    try:
        ApiSettings(
            project_root=tmp_path,
            api_db=tmp_path / "api.sqlite",
            environment="production",
            include_test_evidence=True,
        )
    except RuntimeError as exc:
        assert "Test evidence" in str(exc)
    else:
        raise AssertionError("production test evidence must be rejected")
    assert ApiSettings(project_root=tmp_path).include_test_evidence is False


def test_relative_api_db_env_is_resolved_from_project_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WADDEHHA_API_DB", "runtime/local-api.sqlite3")
    settings = ApiSettings(project_root=tmp_path)
    assert settings.api_db == (tmp_path / "runtime" / "local-api.sqlite3").resolve()


def test_rbac_and_cafe_isolation_for_all_three_roles(client: TestClient, login) -> None:
    owner = login("owner")
    cafes = client.get("/api/cafes").json()["items"]
    assert len(cafes) >= 2
    assert set(owner["cafe_ids"]) <= {cafe["id"] for cafe in cafes}

    client.post("/api/auth/logout")
    manager = login("manager")
    assert client.post(
        "/api/cafes",
        json={
            "name": "Not Allowed",
            "city": "Riyadh",
            "region": "Riyadh",
            "country": "Saudi Arabia",
            "timezone": "Asia/Riyadh",
            "currency": "SAR",
            "seats": 12,
        },
    ).status_code == 403
    assert manager["cafe_ids"]

    client.post("/api/auth/logout")
    employee = login("employee")
    visible = client.get("/api/cafes").json()["items"]
    assert [cafe["id"] for cafe in visible] == employee["cafe_ids"]
    allowed_id = employee["cafe_ids"][0]
    denied_id = next(cafe["id"] for cafe in cafes if cafe["id"] != allowed_id)
    assert client.get(f"/api/cafes/{denied_id}").status_code == 403
    assert client.get(f"/api/cafes/{allowed_id}/data/pos").status_code == 403


def test_real_saved_runs_findings_and_report_are_path_safe(client: TestClient, login) -> None:
    login("owner")
    cafes = client.get("/api/cafes").json()["items"]
    cafe = next(cafe for cafe in cafes if cafe["data_status"] == "available")
    runs_response = client.get(f"/api/runs?cafe_id={cafe['id']}")
    assert runs_response.status_code == 200
    runs = runs_response.json()["items"]
    saved = next(run for run in runs if run["findings_count"] > 0)

    findings = client.get(f"/api/runs/{saved['id']}/findings")
    assert findings.status_code == 200
    assert findings.json()["items"]
    evidence = findings.json()["items"][0]["evidence"][0]
    assert evidence["artifact_id"].startswith("artifact-")
    assert "result_path" not in json.dumps(findings.json())
    assert_no_path_leak(findings.json())

    report = client.get(f"/api/runs/{saved['id']}/report")
    assert report.status_code == 200
    assert report.json()["format"] == "html"
    assert "<html" in report.json()["html"].lower()
    assert_no_path_leak(report.json())


def test_sources_data_pagination_and_lineage_use_real_artifacts(client: TestClient, login) -> None:
    login("owner")
    cafes = client.get("/api/cafes").json()["items"]
    cafe = next(cafe for cafe in cafes if cafe["data_status"] == "available")
    sources = client.get(f"/api/cafes/{cafe['id']}/sources")
    assert sources.status_code == 200
    pos = next(source for source in sources.json()["items"] if source["id"] == "pos")
    assert pos["raw_rows"] > 0
    assert pos["last_run_id"]

    first = client.get(f"/api/cafes/{cafe['id']}/data/pos?limit=2")
    assert first.status_code == 200
    assert len(first.json()["items"]) == 2
    assert first.json()["next_cursor"]
    second = client.get(
        f"/api/cafes/{cafe['id']}/data/pos?limit=2&cursor={first.json()['next_cursor']}"
    )
    assert second.status_code == 200
    assert first.json()["items"][0]["record_id"] != second.json()["items"][0]["record_id"]

    inventory_page = client.get(f"/api/cafes/{cafe['id']}/data/inventory?limit=1")
    assert inventory_page.status_code == 200
    record_id = inventory_page.json()["items"][0]["record_id"]
    lineage = client.get(f"/api/cafes/{cafe['id']}/data/inventory/{record_id}/lineage")
    assert lineage.status_code == 200
    assert lineage.json()["raw"]
    assert lineage.json()["cleaned"]
    assert lineage.json()["changes"]
    assert_no_path_leak(lineage.json())


def test_sse_exposes_checkpoint_history_and_closes_saved_stream(client: TestClient, login) -> None:
    login("owner")
    runs = client.get("/api/runs").json()["items"]
    paused = next(run for run in runs if run["status"] == "waiting_review")
    response = client.get(f"/api/runs/{paused['id']}/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: run.status" in response.text
    assert "event: end" in response.text
    assert paused["id"] in response.text
    assert_no_path_leak(response.text)


def test_manager_review_then_owner_decision_enforces_rbac_and_order(
    client: TestClient, app, login, monkeypatch
) -> None:
    login("owner")
    paused = next(run for run in client.get("/api/runs").json()["items"] if run["status"] == "waiting_review")
    self_review = client.post(
        f"/api/runs/{paused['id']}/manager-review",
        json={"decision": "submit", "comment": "Owner cannot self-review."},
    )
    assert self_review.status_code == 403

    client.post("/api/auth/logout")
    manager = login("manager")
    assert client.post(
        f"/api/runs/{paused['id']}/decision", json={"decision": "approve"}
    ).status_code == 403

    review = client.post(
        f"/api/runs/{paused['id']}/manager-review",
        json={"decision": "submit", "comment": "Evidence checked."},
    )
    assert review.status_code == 201
    assert review.json()["report_state"] == "owner_review"
    assert review.json()["review"]["reviewer_id"] == manager["id"]
    duplicate_review = client.post(
        f"/api/runs/{paused['id']}/manager-review",
        json={"decision": "submit", "comment": "Duplicate."},
    )
    assert duplicate_review.status_code == 409
    assert duplicate_review.json()["error"]["code"] == "invalid_review_transition"

    client.post("/api/auth/logout")
    login("owner")
    resumed: list[tuple[str, str]] = []
    monkeypatch.setattr(app.state.run_service, "can_resume", lambda run_id: True)
    monkeypatch.setattr(app.state.run_service, "resume", lambda run_id, decision: resumed.append((run_id, decision)))
    decision = client.post(
        f"/api/runs/{paused['id']}/decision",
        json={"decision": "approve", "comment": "Approved for delivery."},
    )
    assert decision.status_code == 201
    assert decision.json()["report_state"] == "approved"
    assert resumed == [(paused["id"], "approve")]
    duplicate_decision = client.post(
        f"/api/runs/{paused['id']}/decision", json={"decision": "reject"}
    )
    assert duplicate_decision.status_code == 409
    assert duplicate_decision.json()["error"]["code"] == "invalid_decision_transition"
    assert resumed == [(paused["id"], "approve")]


def test_chat_is_narrowly_grounded_and_cites_evidence(client: TestClient, login) -> None:
    login("owner")
    run = next(run for run in client.get("/api/runs").json()["items"] if run["findings_count"] > 0)
    conversation = client.post(
        "/api/conversations", json={"cafe_id": run["cafe_id"], "run_id": run["id"]}
    )
    assert conversation.status_code == 201
    conversation_id = conversation.json()["conversation"]["id"]

    grounded = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"message": "Show the findings about row count"},
    )
    assert grounded.status_code == 201
    assistant = grounded.json()["assistant_message"]
    assert assistant["citations"]
    assert assistant["citations"][0]["finding_id"].startswith("F-")
    assert_no_path_leak(grounded.json())

    refused = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"message": "Who will win the football match?"},
    )
    assert refused.status_code == 201
    assert refused.json()["assistant_message"]["citations"] == []
    assert "could not match" in refused.json()["assistant_message"]["content"].lower()


def test_rejected_findings_never_reach_repository_or_chat(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_db = project_root / "outputs" / "test_evidence" / "memory_test.sqlite"
    evidence_dir = tmp_path / "outputs" / "test_evidence"
    evidence_dir.mkdir(parents=True)
    copied_db = evidence_dir / "memory_test.sqlite"
    shutil.copy2(source_db, copied_db)
    with closing(sqlite3.connect(copied_db)) as conn:
        run_id, cafe_id = conn.execute(
            "SELECT run_id,profile_key FROM weekly_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        conn.execute("UPDATE findings_history SET approved=0 WHERE run_id=?", (run_id,))
        conn.commit()

    database = ApiDatabase(tmp_path / "api.sqlite")
    repository = ArtifactRepository(tmp_path, database, include_test_evidence=True)
    assert repository.findings(run_id) == []
    answer, citations, grounded_run_id = grounded_answer(
        repository, cafe_id, run_id, "Show all findings"
    )
    assert citations == []
    assert grounded_run_id == run_id
    assert "no grounded findings" in answer.lower()


def test_report_allowlist_and_cross_platform_path_scrubbing() -> None:
    assert_no_path_leak({"source": r"talesof\_khobar"})
    unsafe = """<!doctype html><html><head><style>body{background:url(file:///tmp/x)}</style></head>
    <body onload=alert(1)><script>alert(1)</script><iframe src='https://evil.test'></iframe>
    <svg onload='alert(1)'><script>alert(2)</script></svg><object data='file:///etc/passwd'></object>
    <p class='finding' onclick='alert(1)'>Safe finding C:\\Users\\name\\secret.txt</p>
    <a href='javascript:alert(1)'>link text</a></body></html>"""
    sanitized = sanitize_report_html(unsafe)
    lowered = sanitized.lower()
    assert "<html" in lowered
    assert "safe finding" in lowered
    assert "link text" in lowered
    for forbidden in ("<script", "<style", "<iframe", "<svg", "<object", "onload", "onclick", "javascript:"):
        assert forbidden not in lowered
    assert_no_path_leak(sanitized)

    scrubbed = strip_path_text(
        "file:///tmp/private.txt | /home/user/private.txt | \\\\server\\share\\private.txt | outputs/reports/x.html"
    )
    assert_no_path_leak(scrubbed)
