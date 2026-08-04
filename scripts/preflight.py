"""Standalone preflight check: validates profile, data files and env vars
without running the graph."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.preflight import run_preflight
from src.config.runtime_config import resolve_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--app-settings", type=Path, default=Path("config/app_settings.yaml"))
    parser.add_argument("--source-registry", type=Path, default=Path("config/source_registry.yaml"))
    args = parser.parse_args()

    config = resolve_runtime_config(
        profile_path=args.profile, data_dir=args.data_dir,
        app_settings_path=args.app_settings, source_registry_path=args.source_registry,
    )
    report = run_preflight(config)
    print(f"Preflight OK: {report.ok}")
    for name, check in report.source_checks.items():
        print(f"  {name}: {check['status']}")
    if report.missing_env:
        print(f"  Missing required env vars: {report.missing_env}")
    if report.missing_optional_env:
        print(f"  Missing optional env vars (features degrade): {report.missing_optional_env}")
    if report.errors:
        print(f"  Errors: {report.errors}")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
