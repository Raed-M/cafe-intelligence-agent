from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from api.security import hash_password, new_session_token, token_digest, verify_password


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('owner','manager','employee')),
    password_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    approval_status TEXT NOT NULL DEFAULT 'approved' CHECK(approval_status IN ('pending','approved','rejected')),
    reviewed_by TEXT,
    reviewed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_cafes (
    user_id TEXT NOT NULL,
    cafe_id TEXT NOT NULL,
    PRIMARY KEY (user_id, cafe_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS api_cafes (
    id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (created_by) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS live_runs (
    id TEXT PRIMARY KEY,
    cafe_id TEXT NOT NULL,
    target_week TEXT,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    error_code TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (created_by) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS manager_reviews (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('submit','request_changes')),
    comment TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (reviewer_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS owner_decisions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('approve','edit','reject')),
    comment TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    cafe_id TEXT NOT NULL,
    run_id TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (created_by) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS ix_runs_cafe ON live_runs(cafe_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_messages_conversation ON messages(conversation_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS ux_manager_submit_run
    ON manager_reviews(run_id) WHERE decision = 'submit';
CREATE UNIQUE INDEX IF NOT EXISTS ux_owner_decision_run ON owner_decisions(run_id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApiDatabase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate_schema(conn)

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        if "approval_status" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN approval_status TEXT NOT NULL DEFAULT 'approved'")
        if "reviewed_by" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN reviewed_by TEXT")
        if "reviewed_at" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN reviewed_at TEXT")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _user(row: sqlite3.Row, cafe_ids: list[str]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "email": row["email"],
            "display_name": row["display_name"],
            "role": row["role"],
            "cafe_ids": cafe_ids,
        }

    def seed_development_users(
        self, accounts: tuple[dict[str, str], ...], cafe_ids: list[str]
    ) -> None:
        now = utc_now()
        with self._write_lock, self._connect() as conn:
            for account in accounts:
                user_id = f"dev-{account['role']}"
                conn.execute(
                    """INSERT INTO users (id,email,display_name,role,password_hash,created_at)
                       VALUES (?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET email=excluded.email,
                       display_name=excluded.display_name,role=excluded.role,
                       password_hash=excluded.password_hash,active=1""",
                    (
                        user_id,
                        account["email"].lower(),
                        account["display_name"],
                        account["role"],
                        hash_password(account["password"]),
                        now,
                    ),
                )
                assigned = cafe_ids if account["role"] in {"owner", "manager"} else cafe_ids[:1]
                conn.execute("DELETE FROM user_cafes WHERE user_id = ?", (user_id,))
                conn.executemany(
                    "INSERT INTO user_cafes (user_id,cafe_id) VALUES (?,?)",
                    [(user_id, cafe_id) for cafe_id in assigned],
                )

    def ensure_local_admin(self, username: str, password: str, cafe_ids: list[str]) -> dict[str, Any]:
        now = utc_now()
        username = username.strip().lower()
        with self._write_lock, self._connect() as conn:
            existing = conn.execute("SELECT id FROM users WHERE email=?", (username,)).fetchone()
            user_id = existing["id"] if existing else "local-admin"
            if existing:
                conn.execute(
                    """UPDATE users SET display_name='Admin',role='owner',password_hash=?,active=1,
                       approval_status='approved',reviewed_by=NULL,reviewed_at=? WHERE id=?""",
                    (hash_password(password), now, user_id),
                )
            else:
                conn.execute(
                    """INSERT INTO users
                       (id,email,display_name,role,password_hash,active,approval_status,reviewed_at,created_at)
                       VALUES (?,?,?,'owner',?,1,'approved',?,?)""",
                    (user_id, username, "Admin", hash_password(password), now, now),
                )
            conn.execute("DELETE FROM user_cafes WHERE user_id=?", (user_id,))
            conn.executemany(
                "INSERT INTO user_cafes (user_id,cafe_id) VALUES (?,?)",
                [(user_id, cafe_id) for cafe_id in cafe_ids],
            )
            row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return self._user(row, cafe_ids)

    @staticmethod
    def _access_request(row: sqlite3.Row, cafe_ids: list[str] | None = None) -> dict[str, Any]:
        return {
            "id": row["id"],
            "email": row["email"],
            "display_name": row["display_name"],
            "requested_role": row["role"],
            "status": row["approval_status"],
            "created_at": row["created_at"],
            "reviewed_at": row["reviewed_at"],
            "cafe_ids": cafe_ids or [],
        }

    def create_access_request(self, *, email: str, display_name: str, password: str, role: str) -> dict[str, Any] | None:
        now = utc_now()
        with self._write_lock, self._connect() as conn:
            existing = conn.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
            if existing and existing["approval_status"] in {"pending", "approved"}:
                return None
            if existing:
                user_id = existing["id"]
                conn.execute(
                    """UPDATE users SET display_name=?,role=?,password_hash=?,active=0,
                       approval_status='pending',reviewed_by=NULL,reviewed_at=NULL,created_at=? WHERE id=?""",
                    (display_name, role, hash_password(password), now, user_id),
                )
                conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
                conn.execute("DELETE FROM user_cafes WHERE user_id=?", (user_id,))
            else:
                user_id = f"usr-{uuid.uuid4().hex[:16]}"
                conn.execute(
                    """INSERT INTO users
                       (id,email,display_name,role,password_hash,active,approval_status,created_at)
                       VALUES (?,?,?,?,?,0,'pending',?)""",
                    (user_id, email.lower(), display_name, role, hash_password(password), now),
                )
            row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return self._access_request(row)

    def list_access_requests(self, status: str = "pending") -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM users WHERE role IN ('manager','employee') AND approval_status=? ORDER BY created_at",
                (status,),
            ).fetchall()
            results = []
            for row in rows:
                cafe_ids = [item[0] for item in conn.execute("SELECT cafe_id FROM user_cafes WHERE user_id=? ORDER BY cafe_id", (row["id"],))]
                results.append(self._access_request(row, cafe_ids))
        return results

    def decide_access_request(self, user_id: str, decision: str, reviewer_id: str, cafe_ids: list[str]) -> dict[str, Any] | None:
        now = utc_now()
        with self._write_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id=? AND role IN ('manager','employee') AND approval_status='pending'",
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            approved = decision == "approve"
            conn.execute(
                "UPDATE users SET active=?,approval_status=?,reviewed_by=?,reviewed_at=? WHERE id=?",
                (1 if approved else 0, "approved" if approved else "rejected", reviewer_id, now, user_id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM user_cafes WHERE user_id=?", (user_id,))
            if approved:
                assigned = cafe_ids if row["role"] == "manager" else cafe_ids[:1]
                conn.executemany(
                    "INSERT INTO user_cafes (user_id,cafe_id) VALUES (?,?)",
                    [(user_id, cafe_id) for cafe_id in assigned],
                )
            else:
                assigned = []
            updated = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return self._access_request(updated, assigned)

    def account_approval_status(self, email: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT approval_status FROM users WHERE email=?", (email.lower(),)).fetchone()
        return row[0] if row else None

    def purge_users_except(self, user_id: str) -> int:
        """Remove other local login accounts while preserving owner-created operational records."""
        with self._write_lock, self._connect() as conn:
            keeper = conn.execute("SELECT id,role FROM users WHERE id=?", (user_id,)).fetchone()
            if keeper is None or keeper["role"] != "owner":
                raise ValueError("The retained account must be an existing owner")
            count = conn.execute("SELECT COUNT(*) FROM users WHERE id<>?", (user_id,)).fetchone()[0]
            conn.execute("UPDATE api_cafes SET created_by=? WHERE created_by<>?", (user_id, user_id))
            conn.execute("UPDATE live_runs SET created_by=? WHERE created_by<>?", (user_id, user_id))
            conn.execute("UPDATE conversations SET created_by=? WHERE created_by<>?", (user_id, user_id))
            conn.execute("DELETE FROM manager_reviews WHERE reviewer_id<>?", (user_id,))
            conn.execute("UPDATE owner_decisions SET owner_id=? WHERE owner_id<>?", (user_id, user_id))
            conn.execute("DELETE FROM sessions WHERE user_id<>?", (user_id,))
            conn.execute("DELETE FROM user_cafes WHERE user_id<>?", (user_id,))
            conn.execute("DELETE FROM users WHERE id<>?", (user_id,))
        return int(count)

    def authenticate(self, email: str, password: str, ttl_seconds: int) -> tuple[dict[str, Any], str] | None:
        with self._write_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ? AND active = 1", (email.lower(),)
            ).fetchone()
            if row is None or not verify_password(password, row["password_hash"]):
                return None
            cafe_ids = [
                item[0]
                for item in conn.execute(
                    "SELECT cafe_id FROM user_cafes WHERE user_id = ? ORDER BY cafe_id", (row["id"],)
                )
            ]
            token = new_session_token()
            now = datetime.now(timezone.utc)
            conn.execute(
                "INSERT INTO sessions (token_hash,user_id,expires_at,created_at) VALUES (?,?,?,?)",
                (
                    token_digest(token),
                    row["id"],
                    (now + timedelta(seconds=ttl_seconds)).isoformat(),
                    now.isoformat(),
                ),
            )
            return self._user(row, cafe_ids), token

    def user_for_session(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        now = utc_now()
        with self._write_lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            row = conn.execute(
                """SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id
                   WHERE s.token_hash=? AND s.expires_at>? AND u.active=1""",
                (token_digest(token), now),
            ).fetchone()
            if row is None:
                return None
            cafe_ids = [
                item[0]
                for item in conn.execute(
                    "SELECT cafe_id FROM user_cafes WHERE user_id=? ORDER BY cafe_id", (row["id"],)
                )
            ]
            return self._user(row, cafe_ids)

    def delete_session(self, token: str | None) -> None:
        if not token:
            return
        with self._write_lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash=?", (token_digest(token),))

    def list_api_cafes(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id,profile_json,created_at FROM api_cafes ORDER BY created_at").fetchall()
        return [{"id": row["id"], **json.loads(row["profile_json"])} for row in rows]

    def create_cafe(self, profile: dict[str, Any], user_id: str) -> dict[str, Any]:
        cafe_id = f"cafe-{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO api_cafes (id,profile_json,created_by,created_at) VALUES (?,?,?,?)",
                (cafe_id, json.dumps(profile, ensure_ascii=False), user_id, now),
            )
            conn.execute("INSERT OR IGNORE INTO user_cafes (user_id,cafe_id) VALUES (?,?)", (user_id, cafe_id))
        return {"id": cafe_id, **profile}

    def create_live_run(self, run_id: str, cafe_id: str, target_week: str | None, user_id: str) -> dict[str, Any]:
        now = utc_now()
        with self._write_lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO live_runs
                   (id,cafe_id,target_week,status,stage,created_by,created_at,updated_at)
                   VALUES (?,?,?,'queued','queued',?,?,?)""",
                (run_id, cafe_id, target_week, user_id, now, now),
            )
        return self.get_live_run(run_id) or {}

    def update_live_run(self, run_id: str, *, status: str, stage: str, error_code: str | None = None) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "UPDATE live_runs SET status=?,stage=?,error_code=?,updated_at=? WHERE id=?",
                (status, stage, error_code, utc_now(), run_id),
            )

    def get_live_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM live_runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_live_runs(self, cafe_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if cafe_id:
                rows = conn.execute(
                    "SELECT * FROM live_runs WHERE cafe_id=? ORDER BY created_at DESC", (cafe_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM live_runs ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def add_review(self, run_id: str, reviewer_id: str, decision: str, comment: str) -> dict[str, Any]:
        review = {
            "id": f"review-{uuid.uuid4().hex[:12]}",
            "run_id": run_id,
            "reviewer_id": reviewer_id,
            "decision": decision,
            "comment": comment,
            "created_at": utc_now(),
        }
        try:
            with self._write_lock, self._connect() as conn:
                conn.execute(
                    """INSERT INTO manager_reviews (id,run_id,reviewer_id,decision,comment,created_at)
                       VALUES (:id,:run_id,:reviewer_id,:decision,:comment,:created_at)""",
                    review,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("manager_review_already_submitted") from exc
        return review

    def latest_review(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM manager_reviews WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def add_decision(self, run_id: str, owner_id: str, decision: str, comment: str | None) -> dict[str, Any]:
        item = {
            "id": f"decision-{uuid.uuid4().hex[:12]}",
            "run_id": run_id,
            "owner_id": owner_id,
            "decision": decision,
            "comment": comment,
            "created_at": utc_now(),
        }
        try:
            with self._write_lock, self._connect() as conn:
                review = conn.execute(
                    """SELECT 1 FROM manager_reviews WHERE run_id=? AND decision='submit'
                       ORDER BY created_at DESC LIMIT 1""",
                    (run_id,),
                ).fetchone()
                if review is None:
                    raise ValueError("manager_review_required")
                conn.execute(
                    """INSERT INTO owner_decisions (id,run_id,owner_id,decision,comment,created_at)
                       VALUES (:id,:run_id,:owner_id,:decision,:comment,:created_at)""",
                    item,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("owner_decision_already_recorded") from exc
        return item

    def latest_decision(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM owner_decisions WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def create_conversation(self, cafe_id: str, run_id: str | None, user_id: str) -> dict[str, Any]:
        now = utc_now()
        item = {
            "id": f"conversation-{uuid.uuid4().hex[:12]}",
            "cafe_id": cafe_id,
            "run_id": run_id,
            "created_by": user_id,
            "created_at": now,
            "updated_at": now,
        }
        with self._write_lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO conversations (id,cafe_id,run_id,created_by,created_at,updated_at)
                   VALUES (:id,:cafe_id,:run_id,:created_by,:created_at,:updated_at)""",
                item,
            )
        return item

    def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE created_by=? ORDER BY updated_at DESC", (user_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id=? AND created_by=?", (conversation_id, user_id)
            ).fetchone()
        return dict(row) if row else None

    def add_message(
        self, conversation_id: str, role: str, content: str, citations: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        item = {
            "id": f"message-{uuid.uuid4().hex[:12]}",
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "created_at": utc_now(),
        }
        with self._write_lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO messages (id,conversation_id,role,content,citations_json,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    item["id"], conversation_id, role, content,
                    json.dumps(item["citations"], ensure_ascii=False), item["created_at"],
                ),
            )
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (item["created_at"], conversation_id))
        return item

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC", (conversation_id,)
            ).fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            msg["citations"] = json.loads(msg.pop("citations_json", "[]"))
            result.append(msg)
        return result
