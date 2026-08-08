from __future__ import annotations

import shutil
import uuid
from datetime import date
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


@pytest.fixture(scope="session")
def paused_run(tmp_path_factory) -> dict:
    """Runs the real graph once, with a fake chat model, to a genuine pause at
    the human gate, and returns the resulting checkpoint DB.

    The API surfaces run status, per-source row counts and the review flow by
    reading LangGraph checkpoints. Without one, `/api/runs` has nothing paused
    and `/api/cafes/{id}/sources` falls back to a file-existence stub whose
    `raw_rows` is None -- which is precisely how these tests failed: they
    depended on a checkpoint DB left behind by whoever ran the app locally.
    Building it here makes them hermetic and costs no API calls.

    Session-scoped: the pipeline run takes ~30s, and every test wants the same
    immutable fixture. Each test gets its own copy so a resume in one cannot
    unpause another.
    """
    from src.config.runtime_config import resolve_runtime_config
    from src.graph.main_graph import build_main_graph
    from src.persistence.checkpointer import build_checkpointer
    from tests.fakes import _FakeChatModel

    root = tmp_path_factory.mktemp("paused_run")
    checkpoint_db = root / "checkpoints.sqlite"

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(
            "src.tools.llm_factory.get_chat_model",
            lambda model_name, temperature=0: _FakeChatModel(),
        )
        monkey.setenv("OPENAI_API_KEY", "sk-test-not-real")

        config = resolve_runtime_config(
            profile_path=PROJECT_ROOT / "data" / "qahwa_saihat" / "cafe_profile.json",
            data_dir=PROJECT_ROOT / "data" / "qahwa_saihat",
            app_settings_path=PROJECT_ROOT / "config" / "app_settings.yaml",
            source_registry_path=PROJECT_ROOT / "config" / "source_registry.yaml",
            target_week=date(2026, 1, 5),
            artifact_root=PROJECT_ROOT / "outputs" / "artifacts",
            checkpoint_db=checkpoint_db,
            memory_db=root / "memory.sqlite",
        )
        run_id = "apitest_" + uuid.uuid4().hex[:8]
        saver = build_checkpointer(checkpoint_db)
        try:
            graph = build_main_graph(checkpointer=saver)
            thread_config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
            graph.invoke(
                {
                    "run_id": run_id, "thread_id": run_id, "config": config,
                    "analysis_period": config.analysis_period,
                    "previous_period": config.previous_period,
                    "trailing_baseline_periods": config.trailing_baseline_periods,
                    "recommendation_period": config.recommendation_period,
                    "critic_round": 0, "content_repair_attempts": 0,
                },
                config=thread_config,
            )
            snapshot = graph.get_state(thread_config)
            assert snapshot.next == ("human_gate",), (
                f"fixture must pause at the human gate, got next={snapshot.next}"
            )
        finally:
            saver.conn.close()
    finally:
        monkey.undo()

    return {"run_id": run_id, "checkpoint_db": checkpoint_db, "profile_key": config.profile_key}


@pytest.fixture
def app(tmp_path: Path, paused_run: dict):
    accounts = tuple(
        {
            "role": role,
            "email": f"{role}@test.local",
            "password": password,
            "display_name": role.title(),
        }
        for role, password in PASSWORDS.items()
    )
    # Per-test copy of the session fixture's checkpoint DB: resuming a run
    # mutates it, and one test's owner-decision must not unpause another's.
    checkpoint_db = tmp_path / "checkpoints.sqlite"
    shutil.copy2(paused_run["checkpoint_db"], checkpoint_db)

    settings = ApiSettings(
        project_root=PROJECT_ROOT,
        api_db=tmp_path / "api.sqlite3",
        checkpoint_db=checkpoint_db,
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
