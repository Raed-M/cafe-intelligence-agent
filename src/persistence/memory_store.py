"""Cross-session long-term memory (Module 8 / plan section 22.2): separate from
the LangGraph checkpoint DB. Cross-run statements ("third week in a row",
"you approved this idea last month") are computed from these tables, not
recalled from model memory.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from src.schemas.content import ContentIdea
from src.schemas.findings import AnalystFinding

SCHEMA = """
CREATE TABLE IF NOT EXISTS weekly_runs (
    run_id TEXT PRIMARY KEY,
    profile_key TEXT NOT NULL,
    cafe_name TEXT NOT NULL,
    analysis_start TEXT NOT NULL,
    analysis_end TEXT NOT NULL,
    recommendation_start TEXT NOT NULL,
    recommendation_end TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    findings_count INTEGER NOT NULL DEFAULT 0,
    critic_rejections INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    quality_json TEXT,
    report_html_path TEXT,
    report_pdf_path TEXT,
    whatsapp_summary TEXT,
    UNIQUE(profile_key, analysis_start, analysis_end)
);

CREATE TABLE IF NOT EXISTS findings_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    analyst_name TEXT NOT NULL,
    title TEXT NOT NULL,
    claim TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    source_names_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    approved INTEGER NOT NULL,
    rejection_reason TEXT,
    content_hash TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES weekly_runs(run_id)
);

CREATE TABLE IF NOT EXISTS content_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    idea_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    hook_ar TEXT NOT NULL,
    hook_en TEXT NOT NULL,
    product_sku TEXT NOT NULL,
    post_date TEXT NOT NULL,
    post_time_local TEXT NOT NULL,
    human_decision TEXT,
    edited INTEGER NOT NULL DEFAULT 0,
    delivered INTEGER NOT NULL DEFAULT 0,
    was_posted INTEGER NOT NULL DEFAULT 0,
    performance_json TEXT,
    content_hash TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES weekly_runs(run_id)
);

CREATE TABLE IF NOT EXISTS metric_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    metric_key TEXT NOT NULL,
    dimension_json TEXT NOT NULL,
    value REAL,
    unit TEXT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    comparable INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (run_id) REFERENCES weekly_runs(run_id)
);

CREATE TABLE IF NOT EXISTS delivery_receipts (
    idempotency_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    report_version INTEGER NOT NULL,
    delivered_at TEXT NOT NULL,
    destination_type TEXT NOT NULL,
    artifact_paths_json TEXT NOT NULL
);
"""


class MemoryStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def record_run(
        self, run_id: str, profile_key: str, cafe_name: str, analysis_period: dict[str, str],
        recommendation_period: dict[str, str], started_at: str, completed_at: str | None,
        status: str, final_findings: list[AnalystFinding], critic_rejections: int,
        total_steps: int, total_tokens: int, cost_usd: float, quality: dict[str, Any],
        report_html_path: str | None, report_pdf_path: str | None, whatsapp_summary: str,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO weekly_runs
            (run_id, profile_key, cafe_name, analysis_start, analysis_end, recommendation_start,
             recommendation_end, started_at, completed_at, status, findings_count, critic_rejections,
             total_steps, total_tokens, cost_usd, quality_json, report_html_path, report_pdf_path, whatsapp_summary)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, profile_key, cafe_name, analysis_period["start"], analysis_period["end"],
             recommendation_period["start"], recommendation_period["end"], started_at, completed_at,
             status, len(final_findings), critic_rejections, total_steps, total_tokens, cost_usd,
             json.dumps(quality, default=str), report_html_path, report_pdf_path, whatsapp_summary),
        )
        for f in final_findings:
            self.conn.execute(
                """INSERT INTO findings_history
                (run_id, finding_id, analyst_name, title, claim, metrics_json, source_names_json,
                 confidence, approved, rejection_reason, content_hash)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, f["finding_id"], f["analyst_name"], f["title"], f["claim"],
                 json.dumps(f["evidence"], default=str), json.dumps(f["source_names"]),
                 f["confidence"], 1, None,
                 hashlib.sha256(f["claim"].encode()).hexdigest()),
            )
            for ev in f["evidence"]:
                self.conn.execute(
                    """INSERT INTO metric_history
                    (run_id, metric_key, dimension_json, value, unit, period_start, period_end, comparable)
                    VALUES (?,?,?,?,?,?,?,1)""",
                    (run_id, ev["result_key"], json.dumps({"analyst": f["analyst_name"]}),
                     ev["value"] if isinstance(ev["value"], (int, float)) else None,
                     ev["unit"], ev["period_start"], ev["period_end"]),
                )
        self.conn.commit()

    def record_content(self, run_id: str, ideas: list[ContentIdea], human_decision: str | None, delivered: bool) -> None:
        for idea in ideas:
            self.conn.execute(
                """INSERT INTO content_history
                (run_id, idea_id, finding_id, hook_ar, hook_en, product_sku, post_date, post_time_local,
                 human_decision, edited, delivered, was_posted, performance_json, content_hash)
                VALUES (?,?,?,?,?,?,?,?,?,0,?,0,NULL,?)""",
                (run_id, idea["idea_id"], idea["finding_id"], idea["hook_ar"], idea["hook_en"],
                 idea["product_sku"], idea["post_date"], idea["post_time_local"], human_decision,
                 int(delivered), hashlib.sha256((idea["hook_en"] + idea["hook_ar"]).encode()).hexdigest()),
            )
        self.conn.commit()

    def record_delivery(self, idempotency_key: str, run_id: str, report_version: int, delivered_at: str,
                          destination_type: str, artifact_paths: dict[str, Any]) -> bool:
        """Returns True if newly recorded, False if this key already existed (idempotent no-op)."""
        existing = self.conn.execute(
            "SELECT 1 FROM delivery_receipts WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if existing:
            return False
        self.conn.execute(
            """INSERT INTO delivery_receipts (idempotency_key, run_id, report_version, delivered_at,
               destination_type, artifact_paths_json) VALUES (?,?,?,?,?,?)""",
            (idempotency_key, run_id, report_version, delivered_at, destination_type, json.dumps(artifact_paths)),
        )
        self.conn.commit()
        return True

    def recent_findings(self, profile_key: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT fh.title, fh.claim, fh.analyst_name FROM findings_history fh
               JOIN weekly_runs wr ON wr.run_id = fh.run_id
               WHERE wr.profile_key = ? ORDER BY fh.id DESC LIMIT ?""",
            (profile_key, limit),
        ).fetchall()
        return [{"title": r[0], "claim": r[1], "analyst_name": r[2]} for r in rows]

    def recent_content(self, profile_key: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT ch.hook_en, ch.human_decision, ch.was_posted FROM content_history ch
               JOIN weekly_runs wr ON wr.run_id = ch.run_id
               WHERE wr.profile_key = ? ORDER BY ch.id DESC LIMIT ?""",
            (profile_key, limit),
        ).fetchall()
        return [{"hook_en": r[0], "human_decision": r[1], "was_posted": bool(r[2])} for r in rows]
