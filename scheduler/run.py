"""Autonomous weekly scheduler (Module 8 / M01).

Runs the graph on a cron trigger read from `config/app_settings.yaml`, in the
cafe's own timezone (from the raw profile, never hardcoded), with:
- `misfire_grace_time` so a missed exact-time fire still runs within a window;
- `coalesce=True` so multiple missed fires collapse into one;
- `max_instances=1` at the APScheduler level;
- an additional file-based `RunLock` as defense-in-depth so an overlapping
  trigger is skipped and logged rather than starting a second concurrent run.
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config.runtime_config import resolve_runtime_config
from src.graph.main_graph import build_main_graph
from src.persistence.checkpointer import build_checkpointer
from src.persistence.run_lock import RunLock, RunLockHeld

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scheduler")


def run_scheduled_cycle(profile_path: Path, data_dir: Path, app_settings_path: Path, source_registry_path: Path) -> None:
    config = resolve_runtime_config(
        profile_path=profile_path, data_dir=data_dir,
        app_settings_path=app_settings_path, source_registry_path=source_registry_path,
    )
    lock = RunLock(Path("db") / f"{config.profile_key}.lock")
    try:
        lock.acquire()
    except RunLockHeld:
        logger.warning("Scheduler trigger skipped: a run for %s is already in progress.", config.profile_key)
        return

    try:
        run_id = f"sched_{uuid.uuid4().hex[:8]}"
        logger.info("Autonomous run starting: run_id=%s profile=%s analysis_period=%s",
                     run_id, config.profile_key, config.analysis_period)
        checkpointer = build_checkpointer(config.checkpoint_db)
        graph = build_main_graph(checkpointer=checkpointer)
        thread_config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
        initial_state = {
            "run_id": run_id, "thread_id": run_id, "config": config,
            "analysis_period": config.analysis_period, "previous_period": config.previous_period,
            "trailing_baseline_periods": config.trailing_baseline_periods,
            "recommendation_period": config.recommendation_period,
            "critic_round": 0, "content_repair_attempts": 0,
        }
        out = graph.invoke(initial_state, config=thread_config)
        snapshot = graph.get_state(thread_config)
        if snapshot.next:
            logger.info("Run %s paused for human review before %s. Report: %s",
                        run_id, snapshot.next, out.get("report", {}).get("html_path"))
        else:
            logger.info("Run %s completed with status=%s", run_id, out.get("run_status"))
    finally:
        lock.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--app-settings", type=Path, default=Path("config/app_settings.yaml"))
    parser.add_argument("--source-registry", type=Path, default=Path("config/source_registry.yaml"))
    parser.add_argument("--run-once", action="store_true", help="Run immediately once instead of scheduling.")
    args = parser.parse_args()

    if args.run_once:
        run_scheduled_cycle(args.profile, args.data_dir, args.app_settings, args.source_registry)
        return

    config = resolve_runtime_config(
        profile_path=args.profile, data_dir=args.data_dir,
        app_settings_path=args.app_settings, source_registry_path=args.source_registry,
    )
    schedule = config.app_settings.schedule
    scheduler = BlockingScheduler(timezone=config.raw_profile.timezone)
    scheduler.add_job(
        run_scheduled_cycle,
        trigger=CronTrigger(
            day_of_week=schedule.day_of_week, hour=schedule.hour, minute=schedule.minute,
            timezone=config.raw_profile.timezone,
        ),
        args=[args.profile, args.data_dir, args.app_settings, args.source_registry],
        id=f"weekly_cycle_{config.profile_key}",
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )
    logger.info("Scheduler started for %s (%s): weekly at %s day_of_week=%s hour=%02d:%02d",
                config.raw_profile.cafe_name, config.raw_profile.timezone,
                schedule.day_of_week, schedule.day_of_week, schedule.hour, schedule.minute)
    scheduler.start()


if __name__ == "__main__":
    main()
