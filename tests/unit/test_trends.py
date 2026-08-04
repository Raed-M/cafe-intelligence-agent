from pathlib import Path

from src.persistence.memory_store import MemoryStore
from src.persistence.trends import compute_metric_streak, content_repetition_notes


def _seed_run(store, run_id, period_end, metric_value):
    store.conn.execute(
        """INSERT INTO weekly_runs (run_id, profile_key, cafe_name, analysis_start, analysis_end,
           recommendation_start, recommendation_end, started_at, status)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (run_id, "profile-1", "Test Cafe", "2026-01-01", period_end, "2026-01-08", "2026-01-15",
         "2026-01-01T00:00:00Z", "succeeded"),
    )
    store.conn.execute(
        """INSERT INTO metric_history (run_id, metric_key, dimension_json, value, unit, period_start, period_end)
           VALUES (?,?,?,?,?,?,?)""",
        (run_id, "net_revenue", "{}", metric_value, "SAR", "2026-01-01", period_end),
    )
    store.conn.commit()


def test_streak_detects_consecutive_decline(tmp_path):
    store = MemoryStore(tmp_path / "mem.sqlite")
    _seed_run(store, "r1", "2026-01-08", 1000)
    _seed_run(store, "r2", "2026-01-15", 900)
    _seed_run(store, "r3", "2026-01-22", 800)
    stmt = compute_metric_streak(store, "profile-1", "net_revenue", 700, "2026-01-29")
    assert stmt is not None
    assert stmt.direction == "decrease"
    assert stmt.consecutive_periods == 3  # 3 consecutive week-over-week declines: 1000->900->800->700
    store.close()


def test_streak_none_when_direction_flips(tmp_path):
    store = MemoryStore(tmp_path / "mem.sqlite")
    _seed_run(store, "r1", "2026-01-08", 1000)
    _seed_run(store, "r2", "2026-01-15", 900)
    stmt = compute_metric_streak(store, "profile-1", "net_revenue", 950, "2026-01-22")
    assert stmt is None  # 1000->900 decrease, 900->950 increase: streak broken, len=1


def test_streak_none_with_no_history(tmp_path):
    store = MemoryStore(tmp_path / "mem.sqlite")
    stmt = compute_metric_streak(store, "profile-1", "net_revenue", 1000, "2026-01-08")
    assert stmt is None
    store.close()


def test_content_repetition_detected(tmp_path):
    store = MemoryStore(tmp_path / "mem.sqlite")
    _seed_run(store, "r1", "2026-01-08", 1000)
    store.conn.execute(
        """INSERT INTO content_history (run_id, idea_id, finding_id, hook_ar, hook_en, product_sku,
           post_date, post_time_local, human_decision, was_posted, content_hash)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        ("r1", "i1", "F1", "a", "Try our iced latte", "ICE-001", "2026-01-09", "18:00", "reject", 0, "hash1"),
    )
    store.conn.commit()
    notes = content_repetition_notes(store, "profile-1", "Try our iced latte")
    assert notes and "reject" in notes[0]
    store.close()
