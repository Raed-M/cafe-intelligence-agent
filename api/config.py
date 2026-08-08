from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class ApiSettings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    environment: str = field(default_factory=lambda: os.getenv("WADDEHHA_ENV", "development").strip().lower())
    api_db: Path | None = field(
        default_factory=lambda: (
            Path(value) if (value := os.getenv("WADDEHHA_API_DB", "").strip()) else None
        )
    )
    checkpoint_db: Path | None = field(
        default_factory=lambda: (
            Path(value) if (value := os.getenv("WADDEHHA_CHECKPOINT_DB", "").strip()) else None
        )
    )
    """LangGraph checkpoint store. Defaults to <project_root>/db/checkpoints.sqlite.

    Configurable because ArtifactRepository (which reads run state, source row
    counts and pause status out of it) and RunService (which writes it) must
    agree on one path -- they previously hardcoded it independently, so any
    deployment moving the DB would have silently shown zero runs."""
    cookie_name: str = "waddehha_session"
    cookie_secure: bool | None = None
    session_ttl_seconds: int = 60 * 60 * 12
    seed_users: bool = field(default_factory=lambda: _bool_env("WADDEHHA_DEV_SEED_USERS"))
    include_test_evidence: bool = field(
        default_factory=lambda: _bool_env("WADDEHHA_INCLUDE_TEST_EVIDENCE", False)
    )
    runs_enabled: bool = field(default_factory=lambda: _bool_env("WADDEHHA_RUNS_ENABLED", True))
    allowed_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            value.strip()
            for value in os.getenv(
                "WADDEHHA_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
            ).split(",")
            if value.strip()
        )
    )
    seed_accounts: tuple[dict[str, str], ...] | None = None
    bootstrap_admin: bool | None = None
    admin_username: str = field(default_factory=lambda: os.getenv("WADDEHHA_ADMIN_USERNAME", "admin").strip().lower())
    admin_password: str = field(default_factory=lambda: os.getenv("WADDEHHA_ADMIN_PASSWORD", "admin"))

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        api_db = Path(self.api_db or self.project_root / "db" / "api.sqlite3")
        if not api_db.is_absolute():
            api_db = self.project_root / api_db
        self.api_db = api_db.resolve()
        checkpoint_db = Path(self.checkpoint_db or self.project_root / "db" / "checkpoints.sqlite")
        if not checkpoint_db.is_absolute():
            checkpoint_db = self.project_root / checkpoint_db
        self.checkpoint_db = checkpoint_db.resolve()
        if self.environment != "development":
            self.cookie_secure = True
        elif self.cookie_secure is None:
            self.cookie_secure = False
        if self.environment != "development" and self.seed_users:
            raise RuntimeError("Development seed users cannot be enabled outside development")
        if self.environment != "development" and self.include_test_evidence:
            raise RuntimeError("Test evidence cannot be exposed outside development")
        if self.bootstrap_admin is None:
            self.bootstrap_admin = self.environment == "development"
        if self.environment != "development" and self.bootstrap_admin:
            raise RuntimeError("The local bootstrap admin cannot be enabled outside development")
        if self.seed_accounts is None and self.seed_users:
            self.seed_accounts = self._seed_accounts_from_environment()

    @staticmethod
    def _seed_accounts_from_environment() -> tuple[dict[str, str], ...]:
        definitions = (
            ("owner", "WADDEHHA_DEV_OWNER_EMAIL", "WADDEHHA_DEV_OWNER_PASSWORD", "Development Owner"),
            ("manager", "WADDEHHA_DEV_MANAGER_EMAIL", "WADDEHHA_DEV_MANAGER_PASSWORD", "Development Manager"),
            ("employee", "WADDEHHA_DEV_EMPLOYEE_EMAIL", "WADDEHHA_DEV_EMPLOYEE_PASSWORD", "Development Employee"),
        )
        accounts: list[dict[str, str]] = []
        missing: list[str] = []
        for role, email_key, password_key, display_name in definitions:
            email = os.getenv(email_key, f"{role}@waddehha.local").strip().lower()
            password = os.getenv(password_key, "")
            if not password:
                missing.append(password_key)
            accounts.append(
                {"role": role, "email": email, "password": password, "display_name": display_name}
            )
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Seed users enabled, but seed passwords are missing: {joined}")
        return tuple(accounts)
