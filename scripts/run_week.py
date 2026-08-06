"""CLI entry point: run one weekly cycle end to end.

Usage:
    python scripts/run_week.py --profile data/qahwa_saihat/cafe_profile.json \
        --data-dir data/qahwa_saihat [--target-week 2026-01-05]

The graph pauses before delivery (HITL). Re-run with --resume-thread-id and
--decision {approve,edit,reject} to continue a paused run.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.config.runtime_config import resolve_runtime_config
from src.graph.main_graph import build_main_graph
from src.persistence.checkpointer import build_checkpointer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--target-week", type=str, default=None)
    parser.add_argument("--resume-thread-id", type=str, default=None)
    parser.add_argument("--decision", choices=["approve", "edit", "reject"], default=None)
    parser.add_argument("--app-settings", type=Path, default=Path("config/app_settings.yaml"))
    parser.add_argument("--source-registry", type=Path, default=Path("config/source_registry.yaml"))
    args = parser.parse_args()

    target_week = date.fromisoformat(args.target_week) if args.target_week else None
    config = resolve_runtime_config(
        profile_path=args.profile, data_dir=args.data_dir,
        app_settings_path=args.app_settings, source_registry_path=args.source_registry,
        target_week=target_week,
    )
    checkpointer = build_checkpointer(config.checkpoint_db)
    graph = build_main_graph(checkpointer=checkpointer)

    run_id = args.resume_thread_id or f"run_{uuid.uuid4().hex[:8]}"
    thread_config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}

    if args.resume_thread_id and args.decision:
        graph.update_state(thread_config, {"human_decision": args.decision})
        out = graph.invoke(None, config=thread_config)
        print(f"Resumed {run_id}: run_status={out.get('run_status')}")
        return

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
        print(f"Run {run_id} paused before {snapshot.next}.")
        print(f"Report: {out.get('report', {}).get('html_path')}")
        print(f"WhatsApp summary:\n{out.get('report', {}).get('whatsapp_summary')}")
        print(f"\nTo approve: python scripts/run_week.py --profile {args.profile} "
              f"--data-dir {args.data_dir} --resume-thread-id {run_id} --decision approve")
    else:
        print(f"Run {run_id} completed: run_status={out.get('run_status')}")


if __name__ == "__main__":
    main()
