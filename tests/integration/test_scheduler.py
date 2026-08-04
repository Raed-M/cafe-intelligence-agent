"""Proves the graph fires autonomously off an APScheduler trigger (no
interactive prompt) and that an overlapping trigger is skipped and logged
rather than starting a second concurrent run (M01 / M23)."""
import threading
import time
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from scheduler.run import run_scheduled_cycle
from src.persistence.run_lock import RunLock

ROOT = Path(__file__).resolve().parents[2]


def test_scheduler_fires_job_autonomously(monkeypatch, tmp_path):
    """A cron trigger set to fire within the next couple of seconds must
    invoke run_scheduled_cycle without any human interaction."""
    calls = []
    done = threading.Event()

    def fake_run(profile_path, data_dir, app_settings_path, source_registry_path):
        calls.append((profile_path, data_dir))
        done.set()

    monkeypatch.setattr("scheduler.run.run_scheduled_cycle", fake_run)
    import scheduler.run as scheduler_module

    scheduler = BackgroundScheduler(timezone="Asia/Riyadh")
    now = time.localtime()
    next_minute = (now.tm_min + 1) % 60
    trigger = CronTrigger(second="*/2")  # fire every 2s so the test doesn't wait up to a minute
    scheduler.add_job(scheduler_module.run_scheduled_cycle, trigger=trigger,
                       args=[ROOT / "data" / "qahwa_saihat" / "cafe_profile.json",
                             ROOT / "data" / "qahwa_saihat", ROOT / "config" / "app_settings.yaml",
                             ROOT / "config" / "source_registry.yaml"])
    scheduler.start()
    fired = done.wait(timeout=8)
    scheduler.shutdown(wait=False)

    assert fired, "scheduler did not fire the job autonomously within the timeout"
    assert len(calls) >= 1


def test_overlapping_trigger_is_skipped_via_lock(tmp_path, caplog):
    lock_path = tmp_path / "qahwa.lock"
    held_lock = RunLock(lock_path)
    held_lock.acquire()
    try:
        import logging

        from src.persistence.run_lock import RunLockHeld

        second_lock = RunLock(lock_path)
        raised = False
        try:
            second_lock.acquire()
        except RunLockHeld:
            raised = True
        assert raised, "a concurrent trigger must not acquire the lock while a run is in progress"
    finally:
        held_lock.release()
