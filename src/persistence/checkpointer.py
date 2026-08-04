"""LangGraph SQLite checkpointer wiring. Enables process-restart resume, HITL
pause/resume and (optionally) time-travel debugging via a stable thread_id.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def build_checkpointer(checkpoint_db: Path) -> SqliteSaver:
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_db), check_same_thread=False)
    return SqliteSaver(conn)
